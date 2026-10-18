"""4D indicator functions.

Each takes the math coordinates x, y, z, w as float tensors of the same shape and
returns a boolean tensor: True where the point is inside the shape.
"""

import torch


def julia_mandelbrot(x, y, z, w, precision):
    """Points inside the 4D Julia-Mandelbrot set.

    Axes: x = Re c, z = Im c, y = Re z0, w = Im z0. With the default alignment
    (x on W, z on H) the screen shows the Mandelbrot c-plane, real axis horizontal.
    """
    cr, ci = x, z
    zr, zi = y.clone(), w.clone()
    inside = torch.ones_like(zr, dtype=torch.bool)
    for it in range(precision):
        nzr = zr * zr - zi * zi + cr
        nzi = 2.0 * zr * zi + ci
        inside &= (nzr * nzr + nzi * nzi) <= 4.0
        zr = torch.where(inside, nzr, zr)
        zi = torch.where(inside, nzi, zi)
        if it % 8 == 7 and not bool(inside.any()):
            break
    return inside


def tesseract(x, y, z, w, precision):
    h = 0.5
    return (x.abs() <= h) & (y.abs() <= h) & (z.abs() <= h) & (w.abs() <= h)


def hypersphere(x, y, z, w, precision):
    return (x * x + y * y + z * z + w * w) <= 0.25


def diamond(x, y, z, w, precision):
    return (x.abs() + y.abs() + z.abs() + w.abs()) <= 0.8


def torus(x, y, z, w, precision):
    big, small = 0.5, 0.15
    r3 = torch.sqrt(x * x + y * y + z * z)
    return ((r3 - big) ** 2 + w * w) <= small * small


def hollow_hypersphere(x, y, z, w, precision):
    d = x * x + y * y + z * z + w * w
    return (d >= 0.35 ** 2) & (d <= 0.5 ** 2)


def hypersphere_minus_tesseract(x, y, z, w, precision):
    sphere = (x * x + y * y + z * z + w * w) <= 0.25
    h = 0.3
    cube = (x.abs() <= h) & (y.abs() <= h) & (z.abs() <= h) & (w.abs() <= h)
    return sphere & ~cube


def sine_field(x, y, z, w, precision):
    return (torch.sin(x) + torch.sin(y) + torch.sin(z) + torch.sin(w)) > 0.0


def gyroid_shell(x, y, z, w, precision):
    field = (
        torch.sin(x) * torch.cos(y)
        + torch.sin(y) * torch.cos(z)
        + torch.sin(z) * torch.cos(w)
        + torch.sin(w) * torch.cos(x)
    )
    return field.abs() < 0.25


INDICATORS = {
    "julia_mandelbrot": julia_mandelbrot,
    "tesseract": tesseract,
    "hypersphere": hypersphere,
    "4D_diamond": diamond,
    "4D_torus": torus,
    "hollow_hypersphere": hollow_hypersphere,
    "hypersphere_minus_tesseract": hypersphere_minus_tesseract,
    "4D_sine_field": sine_field,
    "4D_gyroid_shell": gyroid_shell,
}
