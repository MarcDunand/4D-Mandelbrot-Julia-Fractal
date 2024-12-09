import numpy as np
import cv2
from tqdm import tqdm

def mandelbrot_set(xmin, xmax, ymin, ymax, width=1000, height=1000, max_iter=1000):
    x_values = np.linspace(xmin, xmax, width)
    y_values = np.linspace(ymin, ymax, height)

    mandelbrot_image = np.zeros((height, width), dtype=np.uint16)
    for i, y in enumerate(tqdm(y_values, desc="Computing rows")):
        for j, x in enumerate(x_values):
            c = complex(x, y)
            z = 0+0j
            iteration = 0
            for iteration in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    break
            if abs(z) <= 2:
                iteration = max_iter
            mandelbrot_image[i, j] = iteration
    return mandelbrot_image

if __name__ == "__main__":
    xmin, xmax = -1.7, 0.7
    ymin, ymax = -1.1, 1.1
    width, height = 3000, 3000
    max_iter = 200

    print("Computing Mandelbrot set... (this may take a while)")
    data = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)
    print("Done computing.")

    # Create blank RGBA image (black background, fully opaque)
    colored = np.zeros((height, width, 4), dtype=np.uint8)  # Use float32 for accumulation
    colored[..., 3] = 255  # Fully opaque background initially

    # Line color and alpha
    line_color = (50, 50, 50)  # White lines
    line_alpha = 50  # Alpha for each line (0-255, about 20% opacity)
    step = 8

    # Progress bar for line drawing
    total_points = (height // step) * (width // step)
    with tqdm(total=total_points, desc="Drawing lines") as pbar:
        # Create a single mask for all lines
        mask = np.zeros((height, width), dtype=np.uint8)

        # Iterate over points in the set
        for i in range(0, height, step):
            for j in range(0, width, step):
                if data[i, j] == max_iter:
                    c_real = xmin + j * (xmax - xmin) / width
                    c_imag = ymin + i * (ymax - ymin) / height
                    c = complex(c_real, c_imag)

                    # z1 = c
                    z1_real, z1_imag = c.real, c.imag
                    # z2 = c^2 + c
                    z2 = c * c + c
                    z2_real, z2_imag = z2.real, z2.imag

                    z1_j = int((z1_real - xmin) / (xmax - xmin) * width)
                    z1_i = int((z1_imag - ymin) / (ymax - ymin) * height)
                    z2_j = int((z2_real - xmin) / (xmax - xmin) * width)
                    z2_i = int((z2_imag - ymin) / (ymax - ymin) * height)

                    if 0 <= z2_j < width and 0 <= z2_i < height:
                        end_x = int(z1_j + 0.1 * (z2_j - z1_j))
                        end_y = int(z1_i + 0.1 * (z2_i - z1_i))

                        # Draw the line on the mask
                        cv2.line(mask, (z1_j, z1_i), (end_x, end_y), 255, 1, lineType=cv2.LINE_8)

            # Add RGB and alpha values where the mask is white
            colored[mask == 255, :3] = np.clip(
                colored[mask == 255, :3] + line_color, 0, 255
            ).astype(np.uint8)
            colored[mask == 255, 3] = np.clip(
                colored[mask == 255, 3] + line_alpha, 0, 255
            ).astype(np.uint8)

            # Reset mask for the next batch
            mask.fill(0)

            # Update progress bar
            pbar.update(width // step)


    # Clamp values to valid ranges (0-255)
    colored = np.clip(colored, 0, 255).astype(np.uint8)

    # Save the RGBA image with transparency
    cv2.imwrite("mandelbrot_with_lines2.png", colored)

    # Convert to RGB for display
    display_image = cv2.cvtColor(colored, cv2.COLOR_RGBA2RGB)
    cv2.imshow("Mandelbrot Set with Lines", display_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
