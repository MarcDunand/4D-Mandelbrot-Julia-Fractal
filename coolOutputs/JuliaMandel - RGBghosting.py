import numpy as np
import cv2
from tqdm import tqdm

def mandelbrot_set(xr, crmin, crmax, cimin, cimax, width=1000, height=1000, max_iter=1000):
    """
    Compute a color image based on 24 discrete imaginary steps for x0.
    Each pixel color encodes which of these 24 steps converge.
    """
    x_values = np.linspace(crmin, crmax, width)
    y_values = np.linspace(cimin, cimax, height)
    
    # 24 steps for imaginary component of x0 from -2 to 2
    imag_steps = np.linspace(-0.2, 0.2, 24)
    
    # Create a 3-channel color image
    mandelbrot_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            c = complex(x, y)
            
            color_bits = 0  # 24-bit value to store convergence info
            
            # Test each of the 24 imaginary steps
            for idx, m in enumerate(imag_steps):
                z = complex(xr, m)
                
                # Check divergence
                for iteration in range(max_iter):
                    z = z*z + c
                    if abs(z) > 2:
                        break
                else:
                    # If we never broke out, it converged within max_iter
                    color_bits |= (1 << idx)
            
            # Now color_bits is a 24-bit number encoding convergence at all steps.
            # Extract R, G, B
            B = (color_bits >> 0) & 0xFF
            G = (color_bits >> 8) & 0xFF
            R = (color_bits >> 16) & 0xFF
            
            mandelbrot_image[i, j] = [B, G, R]
    
    return mandelbrot_image

if __name__ == "__main__":
    # Define the region of the complex plane to visualize
    crmin, crmax = -1.7, 0.7
    cimin, cimax = -1.1, 1.1
    xrmin, xrmax = -2, 2

    # Resolution and iteration limit
    time, width, height = 20, 200, 200
    max_iter = 20

    # Boolean to save the video
    save_video = True

    # Set up video writer if save_video is True
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
        out = cv2.VideoWriter('mandelbrot_animation.mp4', fourcc, 60, (width, height))

    t_values = np.linspace(xrmin, xrmax, time)

    with tqdm(total=len(t_values), desc="Rendering animation") as pbar:
        for xr in t_values:
            # Compute the Mandelbrot set for the current starting value of z0
            colored = mandelbrot_set(xr, crmin, crmax, cimin, cimax, width, height, max_iter)
            
            # Display the frame
            cv2.imshow("Mandelbrot Set Animation", colored)

            # Save the frame to the video if save_video is True
            if save_video:
                out.write(colored)

            # Add a small delay to allow the frame to update
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            pbar.update(1)

    # Release the video writer if save_video is True
    if save_video:
        out.release()

    # Close the OpenCV window
    cv2.destroyAllWindows()
