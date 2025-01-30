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
animation_thread = None  # Holds the thread running the animation
current_frame = 0    # Tracks current position in animation
total_frames = 30    # Total number of frames (time_steps)

# Set up logging to record debug information to a file
logging.basicConfig(filename='debug.log', level=logging.DEBUG)


def fractalFrame(xySlice, ymin, ymax, xmin, xmax, t, hmin, hmax, height=1000, width=1000, colorBits=24, max_iter=1000, device='cpu'):
    """
    Compute the Mandelbrot set using a 3D grid for full parallelization.
    """
    x_values = torch.linspace(ymin, ymax, width, device=device)
    y_values = torch.linspace(xmin, xmax, height, device=device)
    h_values = torch.linspace(hmin, hmax, colorBits, device=device)

    x_grid, y_grid, h_grid = torch.meshgrid(x_values, y_values, h_values, indexing="ij")
    t_grid = torch.full_like(x_grid, t, dtype=torch.complex64)
    grids = {"H": h_grid, "X": x_grid, "Y": y_grid, "T": t_grid}

    z = grids[xySlice[0]] + 1j * grids[xySlice[1]]
    c = grids[xySlice[2]] + 1j * grids[xySlice[3]]

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


def animation_loop():
    """Runs the animation, allowing pausing, resuming, and slider control."""
    global is_running, is_paused, current_frame

    is_running = True
    xySlice = input().strip()

    if len(xySlice) != 4:
        print("Invalid input! Please enter exactly 4 characters (e.g., 'XYHT').")
        is_running = False
        return

    print(f"Starting animation with parameterization: {xySlice}")

    t_values = torch.linspace(tmin, tmax, total_frames, device=device)

    with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
        while True:
            if not is_running:
                break  # Stop if user presses "Stop"

            compare_frame = current_frame
            while (is_paused and compare_frame == current_frame) or current_frame >= total_frames:
                cv2.waitKey(100)  # Wait without high CPU usage

            t = t_values[current_frame]  # Get t-value based on slider position
            colored = fractalFrame(xySlice, ymin, ymax, xmin, xmax, t, hmin, hmax, height, width, 24, max_iter, device)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[..., 2] = (colored >> 16) & 0xFF
            rgb_image[..., 1] = (colored >> 8) & 0xFF
            rgb_image[..., 0] = colored & 0xFF

            cv2.imshow("Fractal Animation", rgb_image)
            cv2.waitKey(1)

            # Update slider to reflect current frame
            progress_slider.set(current_frame)

            if not is_paused:
                current_frame += 1  # Move forward one frame

            pbar.update(1)

    cv2.destroyAllWindows()
    is_running = False


# Function to get values from entry fields before starting animation
def update_parameters():
    """Update global parameters from input fields before starting the animation."""
    global ymin, ymax, xmin, xmax, tmin, tmax, hmin, hmax

    try:
        ymin = float(ymin_entry.get())
        ymax = float(ymax_entry.get())
        xmin = float(xmin_entry.get())
        xmax = float(xmax_entry.get())
        tmin = float(tmin_entry.get())
        tmax = float(tmax_entry.get())
        hmin = float(hmin_entry.get())
        hmax = float(hmax_entry.get())
        print("Updated parameters successfully.")
    except ValueError:
        print("Invalid input! Please enter numerical values.")



def start_animation():
    """Starts or resumes the animation in a separate thread."""
    global animation_thread, is_paused, is_running

    update_parameters()  # Get updated values from input fields

    if is_running and is_paused:
        print("Resuming animation...")
        is_paused = False
    elif not is_running:
        print("Starting new animation...")
        is_running = True
        is_paused = False
        animation_thread = threading.Thread(target=animation_loop, daemon=True)
        animation_thread.start()


def pause_animation():
    """Pauses the animation."""
    global is_paused
    print("Pausing animation...")
    is_paused = True


def stop_animation():
    """Stops the animation completely and resets frame counter."""
    global is_running, is_paused, current_frame
    print("Stopping animation...")

    is_running = False
    is_paused = False
    current_frame = 0  # Reset frame counter
    progress_slider.set(0)  # Reset slider

    time.sleep(1)  # Allow time for loop to exit before closing OpenCV
    cv2.destroyAllWindows()


def set_frame(val):
    """Sets the current frame from the slider."""
    global current_frame, is_paused
    if is_paused:
        current_frame = int(float(val))
    else:
        is_paused = True  # Pause while adjusting
        current_frame = int(float(val))
        is_paused = False
    

# Dimensional parameter bounds
ymin, ymax = -1, 1
xmin, xmax = -1, 1
tmin, tmax = -1, 1
hmin, hmax = -1, 1

height, width, time_steps, colorBits = 500, 500, total_frames, 24
max_iter = 30
save_video = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# UI setup
root = tk.Tk()
root.title("Fractal Animation Controller")

# Create input fields for ymin, ymax, xmin, xmax, tmin, tmax, hmin, hmax
tk.Label(root, text="Ymin:").pack()
ymin_entry = tk.Entry(root)
ymin_entry.insert(0, str(ymin))  # Set default value
ymin_entry.pack()

tk.Label(root, text="Ymax:").pack()
ymax_entry = tk.Entry(root)
ymax_entry.insert(0, str(ymax))
ymax_entry.pack()

tk.Label(root, text="Xmin:").pack()
xmin_entry = tk.Entry(root)
xmin_entry.insert(0, str(xmin))
xmin_entry.pack()

tk.Label(root, text="Xmax:").pack()
xmax_entry = tk.Entry(root)
xmax_entry.insert(0, str(xmax))
xmax_entry.pack()

tk.Label(root, text="Tmin:").pack()
tmin_entry = tk.Entry(root)
tmin_entry.insert(0, str(tmin))
tmin_entry.pack()

tk.Label(root, text="Tmax:").pack()
tmax_entry = tk.Entry(root)
tmax_entry.insert(0, str(tmax))
tmax_entry.pack()

tk.Label(root, text="Hmin:").pack()
hmin_entry = tk.Entry(root)
hmin_entry.insert(0, str(hmin))
hmin_entry.pack()

tk.Label(root, text="Hmax:").pack()
hmax_entry = tk.Entry(root)
hmax_entry.insert(0, str(hmax))
hmax_entry.pack()

# Play button
play_button = tk.Button(root, text="Play", command=start_animation, font=("Arial", 12))
play_button.pack(pady=10)

# Pause button
pause_button = tk.Button(root, text="Pause", command=pause_animation, font=("Arial", 12))
pause_button.pack(pady=10)

# Stop button
stop_button = tk.Button(root, text="Stop", command=stop_animation, font=("Arial", 12))
stop_button.pack(pady=10)

# Progress slider
progress_slider = tk.Scale(
    root, from_=0, to=total_frames-1, orient="horizontal", length=400,
    command=set_frame, label="Animation Progress"
)
progress_slider.pack(pady=10)

# Start the Tkinter event loop
root.mainloop()