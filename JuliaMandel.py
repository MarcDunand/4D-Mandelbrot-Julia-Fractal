import torch
import cv2
from tqdm import tqdm
import numpy as np


def mandelbrot_set_torch(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits=24, width=1000, height=1000, max_iter=1000, device='cpu'):
    """
    Compute the Mandelbrot set using PyTorch for GPU acceleration.
    """
    # Generate coordinate grids
    x_values = torch.linspace(crmin, crmax, width, device=device)
    y_values = torch.linspace(cimin, cimax, height, device=device)
    c_values = torch.linspace(ximin, ximax, colorBits, device=device)

    # Create complex plane
    x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="ij")
    c_plane = x_grid + 1j * y_grid  # Complex plane on the GPU

    # Initialize color bits and mask
    mandelbrot_image = torch.zeros((width, height), dtype=torch.int32, device=device)

    # Iterate over the 24 discrete imaginary components of x0
    for k, xi in enumerate(c_values):
        z = torch.full_like(c_plane, xr, dtype=torch.complex64, device=device) + 1j * xi  # Starting value for z
        mask = torch.ones_like(z, dtype=torch.bool, device=device)  # Mask to track points that have not diverged

        for _ in range(max_iter):
            z_next = z[mask]**2 + c_plane[mask]  # Compute the next z values for masked points
            divergence = z_next.abs() > 2
            temp_mask = mask.clone()  # Clone the current mask to avoid in-place memory issues
            temp_mask[mask] = ~divergence  # Update the temporary mask for non-diverging points
            mask = temp_mask  # Assign the updated mask back to the original
            z[mask] = z_next[~divergence]  # Update z only for non-diverging points

        mandelbrot_image += mask.int() << k  # Update the color bits based on the final mask

    return mandelbrot_image.cpu().numpy()  # Move result back to the CPU for visualization


if __name__ == "__main__":
    try:
        # Device setup
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using GPU:", torch.cuda.get_device_name(0))
        else:
            device = torch.device("cpu")
            print("Using CPU")

        # Define the region of the complex plane to visualize
        crmin, crmax = -1.7, 0.7
        cimin, cimax = -1.1, 1.1
        xrmin, xrmax = -2, 2
        ximin, ximax = -0.2, 0.2

        # Resolution and iteration limit
        time, colorBits, width, height = 60, 24, 1280, 1280
        max_iter = 120

        # Boolean to save the video
        save_video = True

        # Set up video writer if save_video is True
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
            out = cv2.VideoWriter('./videoOutput/mandelbrot_animation.mp4', fourcc, 30, (width, height))

        t_values = torch.linspace(xrmin, xrmax, time, device=device)

        with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
            for xr in t_values:
                try:
                    # Compute the Mandelbrot set for the current starting value of z0
                    colored = mandelbrot_set_torch(xr, ximin, ximax, crmin, crmax, cimin, cimax, colorBits, width, height, max_iter, device)

                    # Convert hex codes to RGB format
                    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)  # Create NumPy array
                    rgb_image[..., 2] = (colored >> 16) & 0xFF  # Red channel
                    rgb_image[..., 1] = (colored >> 8) & 0xFF   # Green channel
                    rgb_image[..., 0] = colored & 0xFF          # Blue channel

                    # Display the frame
                    cv2.imshow("Mandelbrot Set Animation", rgb_image)

                    # Save the frame to the video if save_video is True
                    if save_video:
                        out.write(rgb_image)

                    # Free GPU memory
                    torch.cuda.empty_cache()

                except Exception as e:
                    print(f"Error during frame computation: {e}")

                # Add a small delay to allow the frame to update
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                pbar.update(1)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Release the video writer if save_video is True
        if save_video:
            out.release()

        # Close the OpenCV window
        cv2.destroyAllWindows()
