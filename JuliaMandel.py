import torch
import cv2
from tqdm import tqdm  # For progress bar
import numpy as np  # For array operations
import psutil  # For checking system memory
import gc  # For garbage collection
import logging  # For logging events
from threading import Timer  # For watchdog timer
import time  # For measuring execution time
import matplotlib.pyplot as plt  # For colormap visualization
import matplotlib.cm as cm  # For colormaps
import tkinter as tk  #For UI

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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)



    #UI setup
    # Create the main application window
    root = tk.Tk()
    root.title("Fractal Animation Controller")

    # Create a button inside the window
    def start_animation():
        print("Button Pressed! Animation will start...")

    start_button = tk.Button(root, text="Start Animation", command=start_animation, font=("Arial", 12))
    start_button.pack(pady=20)  # Add some padding for spacing

    # Start the Tkinter event loop
    root.mainloop()




    while True:
        # Prompt user to select parameterization type
        print("\nChoose parameterization or 'exit' to quit:")
        xySlice = input().strip()
        if xySlice.lower() == 'exit':
            print("Exiting the program.")
            break

        ymin, ymax = -1, 1
        xmin, xmax = -1, 1
        tmin, tmax = -1, 1
        hmin, hmax = -1, 1

        height, width, time_steps, colorBits = 300, 300, 20, 24
        max_iter = 30
        save_video = True

        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('./videoOutput/outp.mp4', fourcc, 10, (width, height))

        t_values = torch.linspace(tmin, tmax, time_steps, device=device)

        def reset_watchdog():
            global watchdog
            watchdog.cancel()
            watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
            watchdog.start()

        watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
        watchdog.start()

        with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
            for idx, t in enumerate(t_values):
                colored = fractalFrame(xySlice, ymin, ymax, xmin, xmax, t, hmin, hmax, width, height, 24, max_iter, device)

                rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_image[..., 2] = (colored >> 16) & 0xFF
                rgb_image[..., 1] = (colored >> 8) & 0xFF
                rgb_image[..., 0] = colored & 0xFF

                cv2.imshow("Fractal Animation", rgb_image)
                cv2.waitKey(1)

                if save_video:
                    out.write(rgb_image)

                reset_watchdog()
                pbar.update(1)

        if save_video:
            out.release()
        cv2.destroyAllWindows()