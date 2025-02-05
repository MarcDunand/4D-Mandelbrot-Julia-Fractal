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
tRes = 30    # Total number of frames (time_steps)

# Set up logging to record debug information to a file
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


def generateFrame(parameterization, ymin, ymax, xmin, xmax, t, hmin, hmax, height=1000, width=1000, colorBits=24, max_iter=1000, device='cpu'):
    """
    Compute the Mandelbrot set using a 3D grid for full parallelization.
    """
    ymin, ymax = ymax*-1, ymin*-1  #needed flip on the y axis
    y_values = torch.linspace(ymin, ymax, height, device=device)
    x_values = torch.linspace(xmin, xmax, width, device=device)
    h_values = torch.linspace(hmin, hmax, colorBits, device=device)

    y_grid, x_grid, h_grid = torch.meshgrid(y_values, x_values, h_values, indexing="ij")
    t_grid = torch.full_like(x_grid, t, dtype=torch.complex64)
    grids = {"H": h_grid, "X": x_grid, "Y": y_grid, "T": t_grid}

    z = grids[parameterization[0]] + 1j * grids[parameterization[1]]
    c = grids[parameterization[2]] + 1j * grids[parameterization[3]]

    mask = torch.ones_like(z, dtype=torch.bool, device=device)

    for _ in range(max_iter):
        z_next = z**2 + c
        divergence = z_next.abs() > 2
        mask = mask & ~divergence
        z = torch.where(mask, z_next, z)

        if not mask.any():
            break

    weights = 2 ** torch.arange(colorBits, device=device, dtype=torch.int32)
    mandelbrot_image = torch.sum(mask.int() * weights, dim=-1)
    return mandelbrot_image.cpu().numpy()


def render_frame():
    t = tmin+(tmax-tmin)*(current_frame/tRes)
    # t_values = torch.linspace(tmin, tmax, tRes, device=device)
    # t = t_values[current_frame]  # Get t-value based on slider position
    colored = generateFrame(parameterization, ymin, ymax, xmin, xmax, t, hmin, hmax, yRes, xRes, hRes, max_iter, device)
    rgb_image = np.zeros((yRes, xRes, 3), dtype=np.uint8)
    rgb_image[..., 2] = (colored >> 16) & 0xFF
    rgb_image[..., 1] = (colored >> 8) & 0xFF
    rgb_image[..., 0] = colored & 0xFF

    cv2.imshow("Fractal Animation", rgb_image)

    cv2.waitKey(1)



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
        while (is_paused and (not needs_update)) or current_frame >= tRes:
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
    global current_frame, parameterization, ymin, ymax, yRes, xmin, xmax, xRes, tmin, tmax, tRes, hmin, hmax, hRes, max_iter

    try:
        tResOld = tRes
        
        parameterization = str(dimen_param.get())

        ymin = float(fields["Y"][0].get())
        ymax = float(fields["Y"][1].get())
        yRes = int(fields["Y"][2].get())
        xmin = float(fields["X"][0].get())
        xmax = float(fields["X"][1].get())
        xRes = int(fields["X"][2].get())
        tmin = float(fields["T"][0].get())
        tmax = float(fields["T"][1].get())
        tRes = int(fields["T"][2].get())
        hmin = float(fields["H"][0].get())
        hmax = float(fields["H"][1].get())
        hRes = int(fields["H"][2].get())

        max_iter = int(prec_param.get())

        if tRes != tResOld:  #if time resolution has been changed, update accordingly
            current_frame = int(current_frame*(tRes/tResOld))  #keeps proportional position in animation
            progress_slider.config(from_=0, to=tRes - 1)  #updates the slider in case the number of frames has changed
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
    global is_paused, needs_update
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


def save_animation():
    update_parameters()  # Ensure parameters are updated
    video_filename = "./videoOutput/fractal_animation.mp4"
    fps = 30  # 30 frames per second

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
    video_writer = cv2.VideoWriter(video_filename, fourcc, fps, (xRes, yRes))
    
    with tqdm(total=tRes, desc="Rendering animation") as pbar:
        for frame_idx in range(tRes):
            global current_frame
            current_frame = frame_idx  # Update current frame index
            
            t = tmin + (tmax - tmin) * (frame_idx / tRes)
            colored = generateFrame(parameterization, ymin, ymax, xmin, xmax, t, hmin, hmax, yRes, xRes, hRes, max_iter, device)
            
            rgb_image = np.zeros((yRes, xRes, 3), dtype=np.uint8)
            rgb_image[..., 2] = (colored >> 16) & 0xFF
            rgb_image[..., 1] = (colored >> 8) & 0xFF
            rgb_image[..., 0] = colored & 0xFF
            
            video_writer.write(rgb_image)  # Add frame to video
            pbar.update(1)  # Update progress bar

    video_writer.release()

    

# Dimensional parameters and their bounds
parameterization = "HTYX"

ymin, ymax = -1, 1
xmin, xmax = -1, 1
tmin, tmax = -1, 1
hmin, hmax = -1, 1

yRes, xRes, hRes = 500, 500, 24
max_iter = 30

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# UI setup
root = tk.Tk()
root.title("Fractal Animation Controller")

# Create a dictionary to store min/max/res labels and entry fields
fields = {
    "Y": None,
    "X": None,
    "T": None,
    "H": None
}



tk.Label(root, text="(Zr + Zi)^2 + (Cr + Ci)").grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="e")
dimen_param = tk.Entry(root)
dimen_param.grid(row=0, column=2, padx=5, pady=5)
dimen_param.insert(0, parameterization)  # Inserts initial values
tk.Label(root, text="(T, Y, X, H)").grid(row=0, column=3, padx=5, pady=5, sticky="e")

tk.Label(root, text="prec").grid(row=0, column=4, padx=5, pady=5, sticky="e")
prec_param = tk.Entry(root)
prec_param.grid(row=0, column=5, padx=5, pady=5)
prec_param.insert(0, max_iter)  # Inserts initial values


for i, key in enumerate(fields.keys()):
    i+=1  #makes space for dimension picker above

    # Labels
    tk.Label(root, text=f"{key}min:").grid(row=i, column=0, padx=5, pady=5, sticky="e")
    tk.Label(root, text=f"{key}max:").grid(row=i, column=2, padx=5, pady=5, sticky="e")
    tk.Label(root, text=f"{key}res:").grid(row=i, column=4, padx=5, pady=5, sticky="e")

    # Entry Fields
    min_entry = tk.Entry(root)
    min_entry.grid(row=i, column=1, padx=5, pady=5)
    min_entry.insert(0, str(eval(f"{key.lower()}min")))  # Inserts initial values

    max_entry = tk.Entry(root)
    max_entry.grid(row=i, column=3, padx=5, pady=5)
    max_entry.insert(0, str(eval(f"{key.lower()}max")))

    res_entry = tk.Entry(root)
    res_entry.grid(row=i, column=5, padx=5, pady=5)
    res_entry.insert(0, str(eval(f"{key.lower()}Res")))

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
    root, from_=0, to=tRes-1, orient="horizontal", length=400,
    command=set_frame, label="Animation Progress"
)
progress_slider.grid(row=6, column=0, columnspan=4, padx=5, pady=5)

# Start the Tkinter event loop
root.mainloop()