PERM = [(i % 3, i // 3) for i in range(24)]

def xyz_to_colorhex(x, y, z):
    return (x << 16) | (y << 8) | z


def gray_decode(g: int) -> int:
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def _deinterleave3(m: int, k: int) -> tuple[int, int, int]:
    a = b = c = 0
    for i in range(k):
        a |= ((m >> (3 * i))     & 1) << i
        b |= ((m >> (3 * i + 1)) & 1) << i
        c |= ((m >> (3 * i + 2)) & 1) << i
    return a, b, c


def key_to_rgb_whiter(key: int) -> tuple[int, int, int]:
    """
    Permute the 24 bits of `key` so that turning *more* bits on makes the
    colour *brighter on average* (closer to white).

    Bijective: every 24-bit key maps to exactly one colour.
    """
    key &= (1 << 24) - 1             # keep exactly 24 bits
    r = g = b = 0
    for i in range(24):
        if key & (1 << i):           # if the i-th bit of the key is 1 …
            ch, pos = PERM[i]        # … find its RGB destination slot
            if ch == 0:              # Red
                r |= 1 << pos
            elif ch == 1:            # Green
                g |= 1 << pos
            else:                    # Blue
                b |= 1 << pos
    return r, g, b


def index_to_color(index: int, bits: int) -> tuple[int, int, int]:
    """
    • Spatial use-case (Morton+Gray) stays identical.
    • Colour now comes **directly from the raw index bits**, shuffled so that
      more 1-bits -> brighter (whiter) RGB, while remaining bijective.
    """
    # ---- 2a.  Brightness-biased colour  (uses *raw* index) ------------
    r, g, b = key_to_rgb_whiter(index)

    # ---- 2b.  (optional) still get the 3-D coordinates if you need them
    #          nothing changes here; this part is independent of colour
    if bits % 3 != 0:
        raise ValueError("bits must be a multiple of 3")
    k = bits // 3
    gx, gy, gz = _deinterleave3(index, k)   # <─ still the *Gray* data
    x = gray_decode(gx)
    y = gray_decode(gy)
    z = gray_decode(gz)
    #   (x,y,z) available here if your application needs it

    # ---- 2c.  Pack colour as 0xRRGGBB and return ----------------------
    return (r << 16) | (g << 8) | b



# n = (1 << 24) - 1
# print(f'{n:08b}')
# print(f'{index_to_color(n, 24):08b}')