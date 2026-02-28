import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, FloatSlider

def generate_fractal(z0_real=0.0, z0_imag=0.0, 
                     x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5, 
                     width=300, height=200, max_iter=100):
    """
    Generate a fractal image based on iteration z_{n+1} = z_n^2 + c,
    starting from z_0 = z0_real + z0_imag*i, and classifying c as before.

    This is similar to Mandelbrot visualization, but with custom initial z0.
    For z0=0, it's the classic Mandelbrot set.
    """
    # Prepare arrays for the image
    mandelbrot_image = np.zeros((height, width))
    x_values = np.linspace(x_min, x_max, width)
    y_values = np.linspace(y_min, y_max, height)
    
    z0 = complex(z0_real, z0_imag)

    for i, y in enumerate(y_values):
        for j, x in enumerate(x_values):
            c = complex(x, y)
            z = z0
            iteration = 0
            for iteration in range(max_iter):
                z = z*z + c
                if abs(z) > 2:
                    break
            mandelbrot_image[i, j] = iteration
    
    return mandelbrot_image

def plot_fractal(z0_real=0.0, z0_imag=0.0):
    # Adjust plot parameters or iteration count as desired
    img = generate_fractal(z0_real=z0_real, z0_imag=z0_imag, 
                           x_min=-2.0, x_max=1.0, y_min=-1.5, y_max=1.5,
                           width=300, height=200, max_iter=100)
    plt.figure(figsize=(6,4))
    plt.imshow(img, extent=(-2,1,-1.5,1.5), origin='lower', cmap='magma')
    plt.title(f"Fractal with z0 = {z0_real} + {z0_imag}i")
    plt.xlabel('Real axis')
    plt.ylabel('Imaginary axis')
    plt.colorbar(label='Iteration count')
    plt.show()

# Create interactive sliders to control the initial z0
interact(plot_fractal, 
         z0_real=FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.01, description='z0_real'),
         z0_imag=FloatSlider(value=0.0, min=-1.0, max=1.0, step=0.01, description='z0_imag'));
