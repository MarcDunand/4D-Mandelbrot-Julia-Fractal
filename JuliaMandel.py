import torch
import cv2
from tqdm import tqdm  # For progress bar
import numpy as np  # For array operations
import psutil  # For checking system memory
import gc  # For garbage collection
import logging  # For logging events
import matplotlib.pyplot as plt  # For colormap visualization
import matplotlib.cm as cm  # For colormaps
import tkinter as tk  #For UI
import threading  # To run animation without freezing UI
import time  # For delays and timing



#globals

# Animation control variables
is_running = False  # True if animation is currently playing
is_paused = False   # True if animation is paused
animation_thread = None  # Holds the thread running the animation



# Set up logging to record debug information to a file
logging.basicConfig(filename='debug.log', level=logging.DEBUG)




def check_memory():
    """
    Monitor system memory usage (CPU and GPU). 
    If usage is high, attempt to free resources.
    """
    memory = psutil.virtual_memory()
    logging.info(f"CPU memory usage: {memory.percent}%")
    if memory.percent > 90:
        gc.collect()  # Trigger garbage collection if CPU memory is over 90%

    # Check GPU memory usage
    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(device=device) / torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(device=device) / 1e6  # Convert to MB
        logging.info(f"GPU memory reserved: {reserved * 100:.2f}%")
        logging.info(f"GPU memory allocated: {allocated:.2f} MB")
        if reserved > 0.9:  # If GPU memory usage exceeds 90%, clear cache
            torch.cuda.empty_cache()


def fractalFrame(xySlice, ymin, ymax, xmin, xmax, t, hmin, hmax, height=1000, width=1000, colorBits=24, max_iter=1000, device='cpu'):
    """
    Compute the Mandelbrot set using a 3D grid for full parallelization.
    """
    x_values = torch.linspace(ymin, ymax, width, device=device)
    y_values = torch.linspace(xmin, xmax, height, device=device)
    h_values = torch.linspace(hmin, hmax, colorBits, device=device)

    x_grid, y_grid, h_grid = torch.meshgrid(x_values, y_values, h_values, indexing="ij")
    t_grid = torch.full_like(x_grid, t, dtype=torch.complex64)  # Placeholder for t with the same shape
    grids = {"H": h_grid, "X": x_grid, "Y": y_grid, "T": t_grid}

    z = grids[xySlice[0]] + 1j * grids[xySlice[1]]
    c = grids[xySlice[2]] + 1j * grids[xySlice[3]]

    mask = torch.ones_like(z, dtype=torch.bool, device=device)  # Shape: [width, height, colorBits]

    for _ in range(max_iter):
        z_next = z**2 + c
        divergence = z_next.abs() > 2  # Identify diverging points
        mask = mask & ~divergence  # Update mask to exclude diverging points
        z = torch.where(mask, z_next, z)  # Update z only for non-diverging points

        if not mask.any():  # Stop if all points diverge
            break

    weights = 2 ** torch.arange(colorBits, device=device, dtype=torch.int32)  # Shape: [colorBits]
    mandelbrot_image = torch.sum(mask.int() * weights, dim=-1)  # Collapse the third dimension

    return mandelbrot_image.cpu().numpy()  # Move result to the CPU for visualization


if __name__ == "__main__":


    def animation_loop():
        global is_running, is_paused

        is_running = True
        xySlice = input().strip()

        if len(xySlice) != 4:
            print("Invalid input! Please enter exactly 4 characters (e.g., 'XYHT').")
            is_running = False
            return

        print(f"Starting animation with parameterization: {xySlice}")
            

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./videoOutput/outp.mp4', fourcc, 10, (width, height))

        t_values = torch.linspace(tmin, tmax, time_steps, device=device)

        for idx, t in enumerate(t_values):
            if not is_running:  
                return  # Stop if user presses "Stop"

            while is_paused:  
                cv2.waitKey(100)  # Wait without high CPU usage

            colored = fractalFrame(xySlice, ymin, ymax, xmin, xmax, t, hmin, hmax, height, width, 24, max_iter, device)
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[..., 2] = (colored >> 16) & 0xFF
            rgb_image[..., 1] = (colored >> 8) & 0xFF
            rgb_image[..., 0] = colored & 0xFF

            cv2.imshow("Fractal Animation", rgb_image)
            cv2.waitKey(1)

        if save_video:
            out.release()

        cv2.destroyAllWindows()
        is_running = False




    def pause_animation():
        """Pauses the animation."""
        global is_paused
        print("Pausing animation...")
        is_paused = True

    

    def stop_animation():
        """Stops the animation completely and safely closes OpenCV."""
        global is_running, is_paused
        print("Stopping animation...")

        is_running = False  # Stops the loop in animation
        is_paused = False  # Reset pause state

        # Wait a short moment to allow loop to exit before closing OpenCV
        time.sleep(1)
        
        cv2.destroyAllWindows()  # Now safe to close OpenCV window




    
    def start_animation(): #new version
        """Starts or resumes the animation in a separate thread."""
        global animation_thread, is_paused, is_running

        if is_running and is_paused:
            print("Resuming animation...")
            is_paused = False  # Resume animation
        elif not is_running:
            print("Starting new animation...")
            is_running = True
            is_paused = False
            animation_thread = threading.Thread(target=animation_loop, daemon=True)
            animation_thread.start()  # Run animation in a new thread



    #sets dimensional parameter bounds
    ymin, ymax = -1, 1
    xmin, xmax = -1, 1
    tmin, tmax = -1, 1
    hmin, hmax = -1, 1

    height, width, time_steps, colorBits = 1000, 1000, 30, 24
    max_iter = 60
    save_video = True


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    #UI setup
    # Create the main application window
    root = tk.Tk()
    root.title("Fractal Animation Controller")


    # tkinter GUI
        
    # Play button
    play_button = tk.Button(root, text="Play", command=start_animation, font=("Arial", 12))
    play_button.pack(pady=10)

    # Pause button
    pause_button = tk.Button(root, text="Pause", command=pause_animation, font=("Arial", 12))
    pause_button.pack(pady=10)

    # Stop button
    stop_button = tk.Button(root, text="Stop", command=stop_animation, font=("Arial", 12))
    stop_button.pack(pady=10)



    # Start the Tkinter event loop
    root.mainloop()