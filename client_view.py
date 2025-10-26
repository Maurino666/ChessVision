# client_view.py
# Purpose: Call your FastAPI /infer endpoint, then visualize the original image
#          with board corners (if any) and piece positions overlaid.
#
# Requirements:
#   pip install requests numpy opencv-python matplotlib
#
# Notes:
# - YOLO runs on the ORIGINAL image on the server. The server returns:
#   - board.corners (TL,TR,BL,BR) in original image pixels (if found)
#   - board.homography (image -> board plane 3x3)
#   - pieces[] with squares in standard orientation and FEN letters.
# - This client maps each piece back to the ORIGINAL image:
#   - If homography is present: it transforms the center of the corresponding
#     board cell from board plane -> image via inverse homography.
#   - If homography is missing: it uses a naive 8x8 grid over the whole image.

import requests
import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# Configuration (edit this)
# ----------------------------
SERVER_URL = "http://127.0.0.1:8000/infer"
WHITE_SIDE = "south"  # south|north|west|east
IMAGE_PATH = r"datasets/real_dataset_2/valid/images/3e0c67f38992fe16dfc163f7f5336263_jpg.rf.4a1c5c9f4cfd6b460d9af2d4c7a870bb.jpg"
SAVE_OVERLAY = "overlay_result.jpg"  # set to None to skip saving

# Virtual board size used by the server when building the homography.
# Keep this in sync with the server's BOARD_SIZE (default 800).
BOARD_SIZE = 800
SHOW_CORNERS = False # Whether to draw detected board corners

# ----------------------------
# Helper functions
# ----------------------------

def algebraic_to_row_col(square: str):
    """
    Converts algebraic notation (e.g., 'e4') to (row, col) in STANDARD orientation:
    - row 0 = rank 8, row 7 = rank 1
    - col 0 = file 'a', col 7 = file 'h'
    """
    files = "abcdefgh"
    file_ch = square[0].lower()
    rank_ch = square[1]
    col = files.index(file_ch)
    row = 8 - int(rank_ch)
    return row, col

def fen_indices_to_prerotation(row_fen: int, col_fen: int, white_side: str):
    """
    Maps STANDARD oriented indices (row_fen, col_fen) back to the pre-rotation
    indices used for assignment on the server (grid_img indexing).
    """
    s = 8
    if white_side == "south":
        # No rotation used on server: grid_fen = grid_img
        return row_fen, col_fen
    elif white_side == "north":
        # Server did: grid_fen = rotate180(grid_img)
        return s - 1 - row_fen, s - 1 - col_fen
    elif white_side == "west":
        # Server did: grid_fen = rotate90_ccw(grid_img)
        # From implementation, inverse mapping is:
        # pre_row = fen_col ; pre_col = (s - 1 - fen_row)
        return col_fen, s - 1 - row_fen
    elif white_side == "east":
        # Server did: grid_fen = rotate90_cw(grid_img)
        # Inverse mapping:
        # pre_row = (s - 1 - fen_col) ; pre_col = fen_row
        return s - 1 - col_fen, row_fen
    else:
        raise ValueError("white_side must be one of: south,north,west,east")

def board_cell_center_in_board_plane(pre_row: int, pre_col: int, board_size: int):
    """
    Returns the center (u, v) of the given pre-rotation cell in the board plane.
    Board plane is 0..board_size in both axes, top-left origin.
    """
    cell = board_size / 8.0
    u = (pre_col + 0.5) * cell
    v = (pre_row + 0.5) * cell
    return float(u), float(v)

def apply_homography_point(H: np.ndarray, x: float, y: float, inverse: bool = False):
    """
    Applies a 3x3 homography to a single point (x, y).
    If inverse=True, uses H^{-1}.
    Returns float pixel coordinates.
    """
    if inverse:
        H = np.linalg.inv(H)
    p = np.array([x, y, 1.0], dtype=np.float64)
    q = H @ p
    if q[2] == 0:
        return None, None
    return float(q[0] / q[2]), float(q[1] / q[2])

def draw_corners(img_bgr: np.ndarray, corners: np.ndarray):
    """
    Draw TL, TR, BL, BR corners and the outline polygon.
    corners: shape (4,2) in original image pixels, order TL, TR, BL, BR.
    """
    vis = img_bgr.copy()
    pts = corners.astype(int)
    # Draw polygon
    poly = np.array([pts[0], pts[1], pts[3], pts[2]], dtype=np.int32).reshape(-1, 1, 2)
    cv2.polylines(vis, [poly], isClosed=True, color=(0, 255, 255), thickness=2, lineType=cv2.LINE_AA)
    # Draw corner markers and labels
    labels = ["TL", "TR", "BL", "BR"]
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (0, 255, 255)]
    for i in range(4):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        cv2.circle(vis, (x, y), 5, colors[i], -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, labels[i], (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2, cv2.LINE_AA)
    return vis

def overlay_pieces(
    img_bgr: np.ndarray,
    pieces: list,
    white_side: str,
    homography: np.ndarray = None
):
    """
    Overlays piece markers on the original image:
    - If homography is provided: map board cell centers (board plane) back to image.
    - Otherwise: map via naive 8x8 image grid.
    Returns a copy of the image with overlays.
    """
    vis = img_bgr.copy()
    H, W = vis.shape[:2]

    for p in pieces:
        sq = p["square"]        # e.g., "e4"
        letter = p["piece"]     # FEN letter 'P','n','B',...
        conf = p.get("confidence", 0.0)

        # Convert algebraic to STANDARD indices, then invert rotation to pre-rotation indices
        r_fen, c_fen = algebraic_to_row_col(sq)
        r_pre, c_pre = fen_indices_to_prerotation(r_fen, c_fen, white_side)

        # Compute image coordinates for this cell center
        if homography is not None:
            u, v = board_cell_center_in_board_plane(r_pre, c_pre, BOARD_SIZE)
            x, y = apply_homography_point(np.asarray(homography, dtype=np.float64), u, v, inverse=True)
            if x is None:
                continue
        else:
            # Naive full-image grid
            cell_w = W / 8.0
            cell_h = H / 8.0
            x = (c_pre + 0.5) * cell_w
            y = (r_pre + 0.5) * cell_h

        # Draw a small circle and label
        cv2.circle(vis, (int(x), int(y)), 10, (0, 200, 255), -1, lineType=cv2.LINE_AA)
        label = f"{letter} ({conf:.2f})"
        cv2.putText(vis, label, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 3, cv2.LINE_AA)
        cv2.putText(vis, label, (int(x) + 8, int(y) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return vis


# ----------------------------
# 1) Call the server
# ----------------------------

img_path = Path(IMAGE_PATH)
assert img_path.exists(), f"Image not found: {img_path}"

with img_path.open("rb") as f:
    files = {"file": f}
    params = {"white_side": WHITE_SIDE}
    resp = requests.post(SERVER_URL, params=params, files=files, timeout=60)

print("HTTP", resp.status_code)
resp.raise_for_status()
data = resp.json()

print("Status:", data.get("status"))
print("FEN:", data.get("fen"))
board = data.get("board", {}) or {}
pieces = data.get("pieces", []) or []

# ----------------------------
# 2) Load image and overlay
# ----------------------------
bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
assert bgr is not None, "Failed to read image with OpenCV."

# Draw corners if available
corners = None
if SHOW_CORNERS and board.get("found") and board.get("corners"):
    corners = np.array(board["corners"], dtype=np.float32)  # TL,TR,BL,BR in original pixels
    bgr = draw_corners(bgr, corners)

# Prepare homography (if provided)
H_img_to_board = None
if board.get("homography"):
    H_img_to_board = np.array(board["homography"], dtype=np.float64)

# Overlay pieces
bgr = overlay_pieces(
    img_bgr=bgr,
    pieces=pieces,
    white_side=board.get("white_side", WHITE_SIDE),  # use server echo if present
    homography=H_img_to_board  # None means naive mapping
)

# Save and show
if SAVE_OVERLAY:
    cv2.imwrite(SAVE_OVERLAY, bgr)
    print(f"Saved overlay to: {SAVE_OVERLAY}")

# Convert BGR->RGB for matplotlib display
rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10, 10))
plt.imshow(rgb)
plt.axis("off")
plt.title(f"FEN: {data.get('fen')}")
plt.show()
