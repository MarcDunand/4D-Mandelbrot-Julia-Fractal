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


# The same shapes in GLSL, for the shader engine (gpu.py). Each defines
#     bool inside_shape(vec4 p, int prec)
# ('precision' is a GLSL keyword)
# with p = (x, y, z, w). Keep them in step with the Python versions above.
GLSL = {
    "julia_mandelbrot": """
bool inside_shape(vec4 p, int prec) {
    float cr = p.x, ci = p.z, zr = p.y, zi = p.w;
    for (int it = 0; it < prec; it++) {
        float nzr = zr * zr - zi * zi + cr;
        float nzi = 2.0 * zr * zi + ci;
        if (!(nzr * nzr + nzi * nzi <= 4.0)) return false;  // also false for NaN, as in PyTorch
        zr = nzr;
        zi = nzi;
    }
    return true;
}""",
    "tesseract": """
bool inside_shape(vec4 p, int prec) {
    return all(lessThanEqual(abs(p), vec4(0.5)));
}""",
    "hypersphere": """
bool inside_shape(vec4 p, int prec) {
    return dot(p, p) <= 0.25;
}""",
    "4D_diamond": """
bool inside_shape(vec4 p, int prec) {
    vec4 a = abs(p);
    return a.x + a.y + a.z + a.w <= 0.8;
}""",
    "4D_torus": """
bool inside_shape(vec4 p, int prec) {
    float d = sqrt(p.x * p.x + p.y * p.y + p.z * p.z) - 0.5;
    return d * d + p.w * p.w <= 0.0225;
}""",
    "hollow_hypersphere": """
bool inside_shape(vec4 p, int prec) {
    float d = dot(p, p);
    return d >= 0.1225 && d <= 0.25;
}""",
    "hypersphere_minus_tesseract": """
bool inside_shape(vec4 p, int prec) {
    return dot(p, p) <= 0.25 && !all(lessThanEqual(abs(p), vec4(0.3)));
}""",
    "4D_sine_field": """
bool inside_shape(vec4 p, int prec) {
    vec4 s = sin(p);
    return s.x + s.y + s.z + s.w > 0.0;
}""",
    "4D_gyroid_shell": """
bool inside_shape(vec4 p, int prec) {
    vec4 s = sin(p), c = cos(p);
    return abs(s.x * c.y + s.y * c.z + s.z * c.w + s.w * c.x) < 0.25;
}""",
}
