import torch
import cv2
import os
from tqdm import tqdm  # For progress bar
import numpy as np  # For array operations
import psutil  # For checking system memory
import gc  # For garbage collection
import logging  # For logging events
import matplotlib.pyplot as plt  # For colormap visualization
import matplotlib.cm as cm  # For colormaps
import tkinter as tk  # For UI
import threading  # To run animation without freezing UI
import time  # For delays and timing
import ColorConverter  # Custom color selection script
import ColorInspector  # Custom UI box for viewing color dimension
from joblib import Parallel, delayed  # To optimize generating LUT

# Globals
is_running = False   # True if animation is currently playing
is_paused = False    # True if animation is paused
needs_update = False  #True if the animation needs to be rerendered
animation_thread = None  # Holds the thread running the animation
current_frame = 0    # Tracks current position in animation
Tres = 30    # Total number of frames (time_steps)


# Runs only once to generate lookup table for use on converting colors
# Runs only when a mapping is requested to generate a LUT
def get_lut(bits: int, mapping_path: str, device='cuda') -> torch.Tensor:
    """
    Build or load a LUT from a mapping .npz.

    bits: number of bits (24)
    mapping_path: path to .npz file with boid_to_rgb and side
    """
    global _lut_cache
    key = (bits, mapping_path)

    if key in _lut_cache:
        return _lut_cache[key]

    # LUT .pt path: same folder + basename as mapping, but .pt extension
    base, _ = os.path.splitext(mapping_path)
    lut_path = base + ".pt"

    if os.path.exists(lut_path):
        print(f"Loading LUT from disk: {lut_path}")
        lut = torch.load(lut_path, map_location=device)
    else:
        print(f"Generating LUT from mapping: {mapping_path}")
        import numpy as np

        data = np.load(mapping_path, allow_pickle=True)
        boid_to_rgb = data["boid_to_rgb"]  # shape (N,3)
        side = int(data["side"])           # cube side
        N = boid_to_rgb.shape[0]
        capacity = 1 << bits

        if N != capacity:
            raise ValueError(
                f"Mapping size N={N} does not match 2**bits={capacity} for bits={bits}"
            )

        x = boid_to_rgb[:, 0].astype(np.int64)
        y = boid_to_rgb[:, 1].astype(np.int64)
        z = boid_to_rgb[:, 2].astype(np.int64)

        # Map voxel coords [0, side-1] to display bytes [0,255]
        if side == 256:
            R = x
            G = y
            B = z
        else:
            # general case
            denom = max(1, side - 1)
            R = (x * 255) // denom
            G = (y * 255) // denom
            B = (z * 255) // denom

        lut_np = ((R << 16) | (G << 8) | B).astype(np.int32)
        lut = torch.from_numpy(lut_np).to(device)

        torch.save(lut.cpu(), lut_path)
        print(f"LUT saved to: {lut_path}")

    _lut_cache[key] = lut
    return lut


def generateFrame(parameterization, Ymin, Ymax, Xmin, Xmax, t, Hmin, Hmax, height=1000, width=1000, colorBits=24, prec=1000, device='cpu'):
    """
    Compute the Mandelbrot set using a 3D grid for full parallelization.
    """
    global XYrot, XTrot, XHrot, YTrot, YHrot, HTrot, lut
    
    rotational_planes = {"XY": XYrot, "XT": XTrot, "XH": XHrot, "YT": YTrot, "YH": YHrot, "HT": HTrot}

    Ymin, Ymax = Ymax*-1, Ymin*-1  #needed flip on the y axis
    y_values = torch.linspace(Ymin, Ymax, height, device=device)
    x_values = torch.linspace(Xmin, Xmax, width, device=device)
    h_values = torch.linspace(Hmin, Hmax, colorBits, device=device)

    y_grid, x_grid, h_grid = torch.meshgrid(y_values, x_values, h_values, indexing="ij")
    t_grid = torch.full_like(x_grid, t, dtype=torch.complex64)
    grids = {"H": h_grid, "X": x_grid, "Y": y_grid, "T": t_grid}

    

    for plane, rot in rotational_planes.items():  #number of cardinal planes in 4D space
        if rot > 0.001:
            axis1 = plane[0]
            axis2 = plane[1]

            # Ensure rotation is a tensor
            tensorRot = torch.tensor(rot, device=device)

            # Compute cosine and sine of the rotation angle
            cosT, sinT = torch.cos(tensorRot), torch.sin(tensorRot)

            # Store original X values before modification
            axis1_original = grids[axis1].clone()

            # Correctly apply the 2D rotation transformation
            grids[axis1] = axis1_original * cosT - torch.real(grids[axis2]) * sinT
            grids[axis2] = axis1_original * sinT + torch.real(grids[axis2]) * cosT



    z = grids[parameterization[0]] + 1j * grids[parameterization[1]]
    c = grids[parameterization[2]] + 1j * grids[parameterization[3]]

    mask = torch.ones_like(z, dtype=torch.bool, device=device)

    for _ in range(prec):
        z_next = z**2 + c
        divergence = z_next.abs() > 2
        mask = mask & ~divergence
        z = torch.where(mask, z_next, z)

        if not mask.any():
            break

    weights = 2 ** torch.arange(colorBits, device=device, dtype=torch.int32)
    mandelbrot_image = torch.sum(mask.int() * weights, dim=-1)

    # If we have a LUT (mapping mode), apply it; otherwise, just return raw indices
    if lut is not None:
        mandelbrot_recolored = lut[mandelbrot_image]
        return mandelbrot_image.cpu().numpy(), mandelbrot_recolored.cpu().numpy()
    else:
        return mandelbrot_image.cpu().numpy(), None


def render_frame(t=None):
    if t is None:
        t = Tmin + (Tmax - Tmin) * (current_frame / Tres)

    index_img, lut_img = generateFrame(
        parameterization, Ymin, Ymax, Xmin, Xmax, t,
        Hmin, Hmax, Yres, Xres, Hres, prec, device
    )

    # Base image: interpret the 24-bit index as 0xRRGGBB
    rgb_image = np.zeros((Yres, Xres, 3), dtype=np.uint8)
    rgb_image[..., 2] = (index_img >> 16) & 0xFF
    rgb_image[..., 1] = (index_img >> 8) & 0xFF
    rgb_image[..., 0] = index_img & 0xFF

    # Cache for re-drawing overlay while paused
    global last_index_img, last_rgb_frame
    last_index_img = index_img
    last_rgb_frame = rgb_image

    # Make a copy for overlay so the returned image stays "clean"
    frame = rgb_image.copy()

    # Draw the hover dialog showing the color-dimension column
    frame = ColorInspector.draw_overlay(frame, index_img, bits=Hres)

    # Show the main frame with overlay
    cv2.imshow("Fractal Animation", frame)

    # Lazily attach mouse callback once the window actually exists
    global inspector_initialized
    if not inspector_initialized:
        ColorInspector.init("Fractal Animation")
        inspector_initialized = True

    # Optional LUT image: only if mapping/LUT is active
    if lut_img is not None:
        lut_image = np.zeros((Yres, Xres, 3), dtype=np.uint8)
        lut_image[..., 2] = (lut_img >> 16) & 0xFF
        lut_image[..., 1] = (lut_img >> 8) & 0xFF
        lut_image[..., 0] = lut_img & 0xFF
        cv2.imshow("Fractal Animation_LUT", lut_image)
    else:
        try:
            cv2.destroyWindow("Fractal Animation_LUT")
        except cv2.error:
            pass
    # For saving to video etc, we keep returning the raw RGB (no overlay)

    return rgb_image

def refresh_overlay():
    """
    Re-draw the current overlay (hover box) on the last rendered frame
    without recomputing the fractal.
    """
    global last_index_img, last_rgb_frame, inspector_initialized

    if last_index_img is None or last_rgb_frame is None:
        return

    frame = last_rgb_frame.copy()
    frame = ColorInspector.draw_overlay(frame, last_index_img, bits=Hres)
    cv2.imshow("Fractal Animation", frame)

    # Make sure the inspector is attached once the window exists
    if not inspector_initialized:
        ColorInspector.init("Fractal Animation")
        inspector_initialized = True




def animation_loop():
    """Runs the animation, allowing pausing, resuming, and slider control."""
    global is_running, is_paused, current_frame, needs_update

    is_running = True

    while True:
        if not is_running:
            break

        # Pause if user presses "pause" or at end of animation
        while (is_paused and (not needs_update)) or current_frame >= Tres:
            # Repaint overlay on the last frame to keep hover dialogue box responsive
            refresh_overlay()
            
            # Pump OpenCV events + allow overlay toggle even while paused
            key = cv2.waitKey(30) & 0xFF
            ColorInspector.handle_key(key)
            if key == 27:  # ESC to stop animation
                is_running = False
                break

        if not is_running:
            break

        # Generate and show the frame
        render_frame()

        needs_update = False

        # Update slider to reflect current frame
        progress_slider.set(current_frame)

        # Handle keyboard input once per frame
        key = cv2.waitKey(1) & 0xFF
        ColorInspector.handle_key(key)
        if key == 27:  # ESC
            is_running = False
            break

        # Move forward one frame
        if not is_paused:
            current_frame += 1

    cv2.destroyAllWindows()
    is_running = False



# Function to get values from entry fields before starting animation
def update_parameters():
    """Update global parameters from input fields before starting the animation."""
    global current_frame, parameterization, Ymin, Ymax, Yres, Xmin, Xmax, Xres, Tmin, Tmax, Tres, Hmin, Hmax, Hres, prec

    try:
        TresOld = Tres
        
        parameterization = str(dimen_param.get())

        Ymin = float(fields["Y"][0].get())
        Ymax = float(fields["Y"][1].get())
        Yres = int(fields["Y"][2].get())
        Xmin = float(fields["X"][0].get())
        Xmax = float(fields["X"][1].get())
        Xres = int(fields["X"][2].get())
        Tmin = float(fields["T"][0].get())
        Tmax = float(fields["T"][1].get())
        Tres = int(fields["T"][2].get())
        Hmin = float(fields["H"][0].get())
        Hmax = float(fields["H"][1].get())
        Hres = int(fields["H"][2].get())

        prec = int(prec_param.get())

        if Tres != TresOld:  #if time resolution has been changed, update accordingly
            current_frame = int(current_frame*(Tres/TresOld))  #keeps proportional position in animation
            progress_slider.config(from_=0, to=Tres - 1)  #updates the slider in case the number of frames has changed
            progress_slider.set(current_frame)  #updates the current position of the slider

    except ValueError:
        print("Invalid input!")



def start_animation():
    """Starts or resumes the animation in a separate thread."""
    global animation_thread, is_paused, is_running

    update_parameters()  # Get updated values from input fields

    if is_running and is_paused:
        is_paused = False
    elif not is_running:
        is_running = True
        is_paused = False
        animation_thread = threading.Thread(target=animation_loop, daemon=True)
        animation_thread.start()


def pause_animation():
    """Pauses the animation."""
    global is_paused
    is_paused = True


def update_animation():
    global is_paused, needs_update, rot
    is_paused = True
    update_parameters()  #updates any changed parameters
    needs_update = True
    


def set_frame(val):
    """Sets the current frame from the slider."""
    global current_frame, is_paused, needs_update
    if is_paused:
        current_frame = int(float(val))
        needs_update = True
    else:
        is_paused = True  # Pause while adjusting
        current_frame = int(float(val))
        is_paused = False


def set_XYrot(val):
    global needs_update, XYrot
    XYrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True

def set_XHrot(val):
    global needs_update, XHrot
    XHrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True

def set_XTrot(val):
    global needs_update, XTrot
    XTrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True

def set_YHrot(val):
    global needs_update, YHrot
    YHrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True

def set_YTrot(val):
    global needs_update, YTrot
    YTrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True

def set_HTrot(val):
    global needs_update, HTrot
    HTrot = float(val) * np.pi / 180.0
    if is_paused:
        needs_update = True


def save_animation():
    update_parameters()  # Ensure parameters are updated
    video_filename = "./videoOutput/fractal_animation.mp4"
    fps = 30  # 30 frames per second

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (Xres, Yres))
    
    with tqdm(total=Tres, desc="Rendering animation") as pbar:
        for frame_idx in range(Tres):
            global current_frame
            current_frame = frame_idx
            
            t = Tmin + (Tmax - Tmin) * (frame_idx / Tres)
            index_img, lut_img = generateFrame(
                parameterization, Ymin, Ymax,
                Xmin, Xmax, t, Hmin, Hmax,
                Yres, Xres, Hres, prec, device
            )

            # Choose what to save: mapped image if available, otherwise base
            if lut_img is not None:
                img_vals = lut_img
            else:
                img_vals = index_img

            rgb_image = np.zeros((Yres, Xres, 3), dtype=np.uint8)
            rgb_image[..., 2] = (img_vals >> 16) & 0xFF
            rgb_image[..., 1] = (img_vals >> 8) & 0xFF
            rgb_image[..., 0] = img_vals & 0xFF
            
            video_writer.write(rgb_image)
            pbar.update(1)

    video_writer.release()



def generateFromCode(initParams, commands, record):
    global device, prec, parameterization, Ymin, Ymax, Yres, Xmin, Xmax, Xres, Hmin, Hmax, Hres, prec, XYrot, XTrot, XHrot, YTrot, YHrot, HTrot 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    parameterization = initParams['parameterization']
    Yres  = initParams['Yres']
    Xres  = initParams['Xres']

    Hres  = int(initParams['Hres'])
    prec  = int(initParams['prec'])
    time  = initParams['time']
    Ymin  = initParams['Ymin']
    Ymax  = initParams['Ymax']
    Xmin  = initParams['Xmin']
    Xmax  = initParams['Xmax']
    Hmin  = initParams['Hmin']
    Hmax  = initParams['Hmax']
    XYrot  = initParams['XYrot']
    XTrot  = initParams['XTrot']
    XHrot  = initParams['XHrot']
    YTrot  = initParams['YTrot']
    YHrot  = initParams['YHrot']
    HTrot  = initParams['HTrot']

    if record:
        video_filename = "./videoOutput/frac_cmded.mp4"
        fps = 30  # 30 frames per second

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
        video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (Xres, Yres))

    with tqdm(total=len(commands), desc="Rendering animation") as pbar:
        for cmd in commands:
            Hres  = int(cmd['Hres'])
            prec  = int(cmd['prec'])
            time  = cmd['time']
            Ymin  = cmd['Ymin']
            Ymax  = cmd['Ymax']
            Xmin  = cmd['Xmin']
            Xmax  = cmd['Xmax']
            Hmin  = cmd['Hmin']
            Hmax  = cmd['Hmax']
            XYrot  = cmd['XYrot'] * np.pi / 180.0
            XTrot  = cmd['XTrot'] * np.pi / 180.0
            XHrot  = cmd['XHrot'] * np.pi / 180.0
            YTrot  = cmd['YTrot'] * np.pi / 180.0
            YHrot  = cmd['YHrot'] * np.pi / 180.0
            HTrot  = cmd['HTrot'] * np.pi / 180.0
            
            if record:
                rgb_image = render_frame(time)
                video_writer.write(rgb_image)
            else:
                render_frame(time)

            pbar.update(1)  # Update progress bar

    if record:
        video_writer.release()

    

# Dimensional parameters and their bounds
parameterization = "HTYX"
XYrot = 0.0
XTrot = 0.0
XHrot = 0.0
YTrot = 0.0
YHrot = 0.0
HTrot = 0.0

Ymin, Ymax = -1, 1
Xmin, Xmax = -1, 1
Tmin, Tmax = -1, 1
Hmin, Hmax = -1, 1

Yres, Xres, Hres = 300, 300, 24
prec = 30

_lut_cache = {}  #Full lookup table for any 24-bit color input


device = None
lut = None
inspector_initialized = False

last_index_img = None
last_rgb_frame = None


fields = {
    "Y": None,
    "X": None,
    "T": None,
    "H": None
}

dimen_param = None
prec_param = None
progress_slider = None



def main(mapping_path=None):
    global device, fields, dimen_param, prec_param, progress_slider, lut

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # If a mapping is provided, build/load its LUT; otherwise, no LUT
    if mapping_path is not None:
        lut = get_lut(24, mapping_path=mapping_path, device=device)
    else:
        lut = None

    # UI setup
    root = tk.Tk()
    root.title("Fractal Animation Controller")

    # Create a dictionary to store min/max/res labels and entry fields

    tk.Label(root, text="(Zr + Zi)^2 + (Cr + Ci)").grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="e")
    dimen_param = tk.Entry(root)
    dimen_param.grid(row=0, column=2, padx=5, pady=5)
    dimen_param.insert(0, parameterization)  # Inserts initial values
    tk.Label(root, text="(T, Y, X, H)").grid(row=0, column=3, padx=5, pady=5, sticky="e")

    tk.Label(root, text="prec").grid(row=0, column=4, padx=5, pady=5, sticky="e")
    prec_param = tk.Entry(root)
    prec_param.grid(row=0, column=5, padx=5, pady=5)
    prec_param.insert(0, prec)  # Inserts initial values


    for i, key in enumerate(fields.keys()):
        i+=1  #makes space for dimension picker above

        # Labels
        tk.Label(root, text=f"{key}min:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
        tk.Label(root, text=f"{key}max:").grid(row=i, column=2, padx=5, pady=5, sticky="e")
        tk.Label(root, text=f"{key}res:").grid(row=i, column=4, padx=5, pady=5, sticky="e")

        # Entry Fields
        min_entry = tk.Entry(root)
        min_entry.grid(row=i, column=1, padx=5, pady=5)
        min_entry.insert(0, str(eval(f"{key}min")))  # Inserts initial values

        max_entry = tk.Entry(root)
        max_entry.grid(row=i, column=3, padx=5, pady=5)
        max_entry.insert(0, str(eval(f"{key}max")))

        res_entry = tk.Entry(root)
        res_entry.grid(row=i, column=5, padx=5, pady=5)
        res_entry.insert(0, str(eval(f"{key}res")))

        # Store in dictionary if needed later
        fields[key] = (min_entry, max_entry, res_entry)

    # Pause button
    pause_button = tk.Button(root, text="Pause", command=pause_animation, font=("Arial", 12))
    pause_button.grid(row=5, column=1, padx=5, pady=10)

    # Play button
    play_button = tk.Button(root, text="Play", command=start_animation, font=("Arial", 12))
    play_button.grid(row=5, column=2, padx=5, pady=10)

    # Update button
    update_button = tk.Button(root, text="Update", command=update_animation, font=("Arial", 12))
    update_button.grid(row=5, column=3, padx=5, pady=10)

    # Save Animation button
    save_button = tk.Button(root, text="Save Animation", command=save_animation, font=("Arial", 12))
    save_button.grid(row=5, column=4, padx=5, pady=10)

    # Progress slider
    progress_slider = tk.Scale(
        root, from_=0, to=Tres-1, orient="horizontal", length=400,
        command=set_frame, label="Animation Progress"
    )
    progress_slider.grid(row=6, column=0, columnspan=4, padx=5, pady=5)

    # Rotation sliders

    tk.Label(root, text="XY rotation:").grid(row=7, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_XYrot
    )
    rotation_slider.grid(row=7, column=1, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="XH rotation:").grid(row=8, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_XHrot
    )
    rotation_slider.grid(row=8, column=1, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="XT rotation:").grid(row=9, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_XTrot
    )
    rotation_slider.grid(row=9, column=1, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="YH rotation:").grid(row=10, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_YHrot
    )
    rotation_slider.grid(row=10, column=1, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="YT rotation:").grid(row=11, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_YTrot
    )
    rotation_slider.grid(row=11, column=1, columnspan=3, padx=5, pady=5)

    tk.Label(root, text="HT rotation:").grid(row=12, column=0, padx=5, sticky="e")
    rotation_slider = tk.Scale(
        root, from_=0, to=359, orient="horizontal", length=360,
        command=set_HTrot
    )
    rotation_slider.grid(row=12, column=1, columnspan=3, padx=5, pady=5)

    # Start the Tkinter event loop
    root.mainloop()



if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mapping",
        type=str,
        default=None,
        help="Optional .npz mapping file. If provided, a corresponding .pt LUT is loaded/created and used."
    )
    args = ap.parse_args()
    main(mapping_path=args.mapping)