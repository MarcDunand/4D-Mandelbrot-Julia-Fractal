#File for hover UI element that shows the color dimension of the
#pixel that you are hovering over in the visualization

import cv2

# Name of the OpenCV window we attach to
WINDOW_NAME = "Fractal Animation"

# Internal state
_hover_x = 0
_hover_y = 0
_hover_inside = False
_overlay_enabled = False


def init(window_name: str = "Fractal Animation") -> None:
    """
    Attach a mouse callback to the given OpenCV window.
    Call this once after cv2.namedWindow(window_name).
    """
    global WINDOW_NAME
    WINDOW_NAME = window_name
    cv2.setMouseCallback(WINDOW_NAME, _on_mouse)


def _on_mouse(event, x, y, flags, param):
    """Track the current mouse position inside the window."""
    global _hover_x, _hover_y, _hover_inside

    if event in (cv2.EVENT_MOUSEMOVE,
                 cv2.EVENT_LBUTTONDOWN,
                 cv2.EVENT_RBUTTONDOWN,
                 cv2.EVENT_MBUTTONDOWN):
        _hover_x = x
        _hover_y = y
        _hover_inside = True
    # OpenCV doesn't give a reliable "leave window" event, so we keep
    # _hover_inside = True once we've seen the mouse here.


def handle_key(key: int) -> None:
    """
    Handle keyboard input. Right now:
      - space toggles the overlay on/off.
    Call this once per frame with the result of cv2.waitKey(...) & 0xFF.
    """
    global _overlay_enabled

    if key == ord(' '):
        _overlay_enabled = not _overlay_enabled


def draw_overlay(frame_bgr, index_img, bits: int = 24):
    """
    Draw a thin single column of `bits` bars showing which positions along
    the color dimension are filled for the pixel under the mouse.

    - frame_bgr: (H, W, 3) uint8, BGR, will be modified in place
    - index_img: (H, W) int, same indexing as your mandelbrot_image
    - bits: how many positions along the color dimension (usually Hres / 24)
    """
    if not _overlay_enabled or not _hover_inside:
        return frame_bgr

    h_idx, w_idx = index_img.shape[:2]
    x, y = _hover_x, _hover_y

    # Ensure the hover point is within the image
    if x < 0 or x >= w_idx or y < 0 or y >= h_idx:
        return frame_bgr

    # Get the packed bit pattern for this pixel
    val = int(index_img[y, x])

    # Decode occupancy bits: bit i = 1 if that color position is filled
    # You can flip this later if you want the semantic of which side is "0".
    occupancy = [(val >> i) & 1 for i in range(bits)]

    img_h, img_w, _ = frame_bgr.shape

    # Layout parameters for a single vertical stack
    bar_width = 12      # px (thin column)
    bar_height = 6      # px per bar
    gap = 2             # px between bars
    padding = 4         # px padding inside box
    margin = 10         # px away from cursor

    total_height = bits * bar_height + (bits - 1) * gap
    box_width = bar_width + 2 * padding
    box_height = total_height + 2 * padding

    # Default: box appears to the right & below the cursor
    box_x = x + margin
    box_y = y + margin

    # Keep box fully on-screen
    if box_x + box_width > img_w:
        box_x = x - margin - box_width
    if box_y + box_height > img_h:
        box_y = y - margin - box_height

    box_x = max(0, box_x)
    box_y = max(0, box_y)

    # Semi-transparent dark background
    overlay = frame_bgr.copy()
    cv2.rectangle(
        overlay,
        (box_x, box_y),
        (box_x + box_width, box_y + box_height),
        (0, 0, 0),
        thickness=-1,
    )
    alpha = 0.6
    cv2.addWeighted(overlay, alpha, frame_bgr, 1 - alpha, 0, dst=frame_bgr)

    # Draw vertical stack of bars, one per bit
    # Here: top bar = highest index (bits-1), bottom = 0
    start_y = box_y + padding
    for i in range(bits):
        bit_index = bits - 1 - i  # flip so highest index at top
        filled = occupancy[bit_index] == 1

        y0 = start_y + i * (bar_height + gap)
        y1 = y0 + bar_height - 1
        x0 = box_x + padding
        x1 = x0 + bar_width - 1

        # Filled vs empty style
        color_fill = (0, 255, 0) if filled else (40, 40, 40)
        color_border = (220, 220, 220)

        if filled:
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_fill, thickness=-1)
        else:
            # thin lines for empty bars
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_fill, thickness=1)

        # Border around every bar
        cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), color_border, thickness=1)

    # Small marker at the sample point
    cv2.circle(frame_bgr, (x, y), 3, (255, 255, 255), thickness=1)

    return frame_bgr
