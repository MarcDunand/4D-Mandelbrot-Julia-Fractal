import numpy as np
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def mandelbrot_row(row_idx, xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter):
    """
    Compute a single row of the Mandelbrot set image.
    """
    x_values = np.linspace(crmin, crmax, width)
    y = np.linspace(cimin, cimax, height)[row_idx]
    c = x_values + y * 1j  # Create complex c for the row

    c_values = np.linspace(ximin, ximax, colorBits)
    row_colors = np.zeros(width, dtype=np.uint32)

    for j, cx in enumerate(c):
        color_bits = 0
        for k, xi in enumerate(c_values):
            z = complex(xr, xi)
            for iteration in range(max_iter):
                z = z * z + cx
                if abs(z) > 2:
                    break
            else:
                color_bits |= (1 << k)
        row_colors[j] = color_bits

    return row_idx, row_colors

def mandelbrot_set_parallel(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter):
    """
    Parallel computation of the Mandelbrot set image.
    """
    rows = np.zeros((height, width), dtype=np.uint32)

    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                mandelbrot_row, i, xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter
            )
            for i in range(height)
        ]

        for future in tqdm(futures, desc="Computing Mandelbrot rows", total=height):
            row_idx, row_colors = future.result()
            rows[row_idx] = row_colors

    return rows

if __name__ == "__main__":
    # Define the region of the complex plane to visualize
    crmin, crmax = -1.7, 0.7
    cimin, cimax = -1.1, 1.1
    xrmin, xrmax = -2, 2
    ximin, ximax = -0.2, 0.2

    # Resolution and iteration limit
    time, colorBits, width, height = 120, 24, 640, 640
    max_iter = 60

    # Boolean to save the video
    save_video = True

    # Set up video writer if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter('./videoOutput/mandelbrot_animation.mp4', fourcc, 30, (width, height))

    t_values = np.linspace(xrmin, xrmax, time)

    with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
        for xr in t_values:
            # Compute the Mandelbrot set for the current starting value of z0
            colored = mandelbrot_set_parallel(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter)
            
            # Convert hex codes to RGB format
            rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
            rgb_image[..., 2] = (colored >> 16) & 0xFF  # Red channel
            rgb_image[..., 1] = (colored >> 8) & 0xFF   # Green channel
            rgb_image[..., 0] = colored & 0xFF          # Blue channel

            # Display the frame
            cv2.imshow("Mandelbrot Set Animation", rgb_image)

            # Save the frame to the video if save_video is True
            if save_video:
                out.write(rgb_image)

            # Add a small delay to allow the frame to update
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            pbar.update(1)

    # Release the video writer if save_video is True
    if save_video:
        out.release()

    # Close the OpenCV window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
