import torch
import cv2
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

# Globals
is_running = False   # True if animation is currently playing
is_paused = False    # True if animation is paused
needs_update = False  #True if the animation needs to be rerendered
animation_thread = None  # Holds the thread running the animation
current_frame = 0    # Tracks current position in animation
Tres = 30    # Total number of frames (time_steps)

def generateFrame(parameterization, Ymin, Ymax, Xmin, Xmax, t, Hmin, Hmax, height=1000, width=1000, colorBits=24, prec=1000, device='cpu'):
    """
    Compute the Mandelbrot set using a 3D grid for full parallelization.
    """
    global XYrot, XTrot, XHrot, YTrot, YHrot, HTrot
    
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
    return mandelbrot_image.cpu().numpy()


def render_frame(t = None):
    if t == None:
        t = Tmin+(Tmax-Tmin)*(current_frame/Tres)
    # t_values = torch.linspace(Tmin, Tmax, Tres, device=device)
    # t = t_values[current_frame]  # Get t-value based on slider position
    colored = generateFrame(parameterization, Ymin, Ymax, Xmin, Xmax, t, Hmin, Hmax, Yres, Xres, Hres, prec, device)
    rgb_image = np.zeros((Yres, Xres, 3), dtype=np.uint8)
    rgb_image[..., 2] = (colored >> 16) & 0xFF
    rgb_image[..., 1] = (colored >> 8) & 0xFF
    rgb_image[..., 0] = colored & 0xFF

    cv2.imshow("Fractal Animation", rgb_image)

    cv2.waitKey(1)

    return rgb_image



def animation_loop():
    """Runs the animation, allowing pausing, resuming, and slider control."""
    global is_running, is_paused, current_frame, needs_update

    is_running = True

    while True:
        # Stop if user presses "Stop"
        if not is_running:
            break
        
        # Pause if user presses "pause" or at end of animation
        #compare_frame = current_frame
        while (is_paused and (not needs_update)) or current_frame >= Tres:
            cv2.waitKey(100)
        
        #Generates and shows the frame
        render_frame()

        needs_update = False

        # Update slider to reflect current frame
        progress_slider.set(current_frame)
        
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

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (Xres, Yres))
    
    with tqdm(total=Tres, desc="Rendering animation") as pbar:
        for frame_idx in range(Tres):
            global current_frame
            current_frame = frame_idx  # Update current frame index
            
            t = Tmin + (Tmax - Tmin) * (frame_idx / Tres)
            colored = generateFrame(parameterization, Ymin, Ymax, Xmin, Xmax, t, Hmin, Hmax, Yres, Xres, Hres, prec, device)
            
            rgb_image = np.zeros((Yres, Xres, 3), dtype=np.uint8)
            rgb_image[..., 2] = (colored >> 16) & 0xFF
            rgb_image[..., 1] = (colored >> 8) & 0xFF
            rgb_image[..., 0] = colored & 0xFF
            
            video_writer.write(rgb_image)  # Add frame to video
            pbar.update(1)  # Update progress bar

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


device = None


fields = {
    "Y": None,
    "X": None,
    "T": None,
    "H": None
}

dimen_param = None
prec_param = None
progress_slider = None



def main():
    global device, fields, dimen_param, prec_param, progress_slider

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

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
    main()