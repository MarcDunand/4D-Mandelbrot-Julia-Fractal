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
import os  # For system operations like exiting

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
    Compute the Mandelbrot set using PyTorch for GPU acceleration.
    Each pixel encodes whether different imaginary components of z0 converge.
    """
    # Create grids of x and y values representing the complex plane
    x_values = torch.linspace(ymin, ymax, width, device=device)
    y_values = torch.linspace(xmin, xmax, height, device=device)
    z0_img_values = torch.linspace(hmin, hmax, colorBits, device=device)
    x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
    
    xy_plane = x_grid + 1j * y_grid  # Represent the complex plane

    # Initialize an image to store convergence data (24-bit color encoding)
    mandelbrot_image = torch.zeros((width, height), dtype=torch.int32, device=device)

    # Iterate over the 24 discrete imaginary components of z0
    for k, xi in enumerate(z0_img_values):
        # Initialize z0 with real component t and current imaginary component xi
        
        if xySlice == "M":
            z = torch.full_like(xy_plane, t, dtype=torch.complex64) + 1j * xi
            c = xy_plane.clone()
        elif xySlice == "J":
            z = xy_plane.clone()
            c = torch.full_like(xy_plane, t, dtype=torch.complex64) + 1j * xi

        mask = torch.ones_like(z, dtype=torch.bool)  # Track points that haven't diverged

        for _ in range(max_iter):
            # Calculate z^2 + c for points that haven't diverged

            z_next = z[mask]**2 + c[mask]

            divergence = z_next.abs() > 2  # Identify diverging points
            temp_mask = mask.clone()
            temp_mask[mask] = ~divergence  # Update mask to exclude diverging points
            mask = temp_mask
            z[mask] = z_next[~divergence]  # Update z for non-diverging points

            if not mask.any():  # Stop if all points diverge
                break

        # Encode the result for this imaginary component into the 24-bit color value
        mandelbrot_image += mask.int() << k

    return mandelbrot_image.cpu().numpy()  # Move the result back to the CPU for visualization

if __name__ == "__main__":
    # Set the device for computation (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Define the region of the complex plane to visualize
    ymin, ymax = -1.05, 1.05
    xmin, xmax = -1.2, 1.2
    tmin, tmax = 0.27, 0.45  # Range for the real component of z0
    hmin, hmax = -0.45, 0.45  # Range for the imaginary component of z0

    # Set parameters for rendering
    height, width, time_steps, colorBits = 300, 390, 10, 24  # Resolution and animation frames, COLORBITS need not be 24
    max_iter = 30  # Maximum iterations for convergence
    save_video = True  # Whether to save the animation as a video

    # Initialize the video writer if saving the video
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
        out = cv2.VideoWriter('./videoOutput/mandelbrot_animation.mp4', fourcc, 10, (width, height))

    # Generate a range of real values for the animation
    t_values = torch.linspace(tmin, tmax, time_steps, device=device)

    # Function to reset the watchdog timer (prevents stalls)
    def reset_watchdog():
        global watchdog
        watchdog.cancel()
        watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
        watchdog.start()

    # Start the watchdog timer
    watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
    watchdog.start()

    # Render the animation
    with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
        for idx, t in enumerate(t_values):
            try:
                start_time = time.time()
                check_memory()  # Check memory usage
                logging.info(f"Starting frame {idx} with t = {t}")

                # Compute the Mandelbrot set for the current frame
                colored = fractalFrame("J", ymin, ymax, xmin, xmax, t, hmin, hmax, width, height, 24, max_iter, device)

                # Convert the Mandelbrot image to RGB format
                rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_image[..., 2] = (colored >> 16) & 0xFF  # Red channel
                rgb_image[..., 1] = (colored >> 8) & 0xFF   # Green channel
                rgb_image[..., 0] = colored & 0xFF          # Blue channel

                # Display the frame
                cv2.imshow("Mandelbrot Set Animation", rgb_image)
                cv2.waitKey(1)

                # Save the frame if video saving is enabled
                if save_video:
                    out.write(rgb_image)

                elapsed_time = time.time() - start_time
                logging.info(f"Frame {idx} completed in {elapsed_time:.2f} seconds.")

                reset_watchdog()  # Reset the watchdog timer

            except Exception as e:
                logging.error(f"Error in frame {idx}: {e}")

            pbar.update(1)  # Update the progress bar

    # Release resources and close the OpenCV window
    if save_video:
        out.release()
    cv2.destroyAllWindows()
    os._exit(0)  # Force the program to exit immediately
