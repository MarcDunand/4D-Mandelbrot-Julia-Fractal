import torch
import cv2
from tqdm import tqdm
import numpy as np
import psutil
import gc
import logging
from threading import Timer
import time

logging.basicConfig(filename='debug.log', level=logging.DEBUG)

def check_memory():
    memory = psutil.virtual_memory()
    logging.info(f"CPU memory usage: {memory.percent}%")
    if memory.percent > 90:
        gc.collect()

    if torch.cuda.is_available():
        reserved = torch.cuda.memory_reserved(device=device) / torch.cuda.get_device_properties(0).total_memory
        allocated = torch.cuda.memory_allocated(device=device) / 1e6
        logging.info(f"GPU memory reserved: {reserved * 100:.2f}%")
        logging.info(f"GPU memory allocated: {allocated:.2f} MB")
        if reserved > 0.9:
            torch.cuda.empty_cache()

def mandelbrot_set_torch(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits=24, width=1000, height=1000, max_iter=1000, device='cpu'):
    x_values = torch.linspace(crmin, crmax, width, device=device)
    y_values = torch.linspace(cimin, cimax, height, device=device)
    c_values = torch.linspace(ximin, ximax, colorBits, device=device)
    x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
    c_plane = x_grid + 1j * y_grid
    mandelbrot_image = torch.zeros((width, height), dtype=torch.int32, device=device)

    for k, xi in enumerate(c_values):
        z = torch.full_like(c_plane, xr, dtype=torch.complex64) + 1j * xi
        mask = torch.ones_like(z, dtype=torch.bool, device=device)
        for _ in range(max_iter):
            z_next = z[mask]**2 + c_plane[mask]
            divergence = z_next.abs() > 2
            temp_mask = mask.clone()
            temp_mask[mask] = ~divergence
            mask = temp_mask
            z[mask] = z_next[~divergence]
            if not mask.any():
                break
        mandelbrot_image += mask.int() << k

    return mandelbrot_image.cpu().numpy()

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    crmin, crmax = -1.7, 0.7
    cimin, cimax = -1.1, 1.1
    xrmin, xrmax = -2, 2
    ximin, ximax = -0.2, 0.2

    time_steps, colorBits, width, height = 30, 24, 300, 300
    max_iter = 30
    save_video = False

    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('./videoOutput/mandelbrot_animation.mp4', fourcc, 30, (width, height))

    t_values = torch.linspace(xrmin, xrmax, time_steps, device=device)

    def reset_watchdog():
        global watchdog
        watchdog.cancel()
        watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
        watchdog.start()

    watchdog = Timer(60, lambda: logging.error("Watchdog timer expired. Program stalled."))
    watchdog.start()

    with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
        for idx, xr in enumerate(t_values):
            try:
                start_time = time.time()
                check_memory()
                logging.info(f"Starting frame {idx} with xr = {xr}")

                colored = mandelbrot_set_torch(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter, device)

                rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
                rgb_image[..., 2] = (colored >> 16) & 0xFF
                rgb_image[..., 1] = (colored >> 8) & 0xFF
                rgb_image[..., 0] = colored & 0xFF

                cv2.imshow("Mandelbrot Set Animation", rgb_image)
                cv2.waitKey(1)

                if save_video:
                    out.write(rgb_image)

                elapsed_time = time.time() - start_time
                logging.info(f"Frame {idx} completed in {elapsed_time:.2f} seconds.")

                reset_watchdog()

            except Exception as e:
                logging.error(f"Error in frame {idx}: {e}")

            pbar.update(1)

    if save_video:
        out.release()
    cv2.destroyAllWindows()
