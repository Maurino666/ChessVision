import numpy as np
import cv2
from typing import Tuple

def get_homography_from_corners(corners_tl_tr_bl_br: np.ndarray, board_size: int) -> np.ndarray:
    """Builds image->board homography to a board_size x board_size square plane."""
    dst = np.array([
        [0, 0],
        [board_size - 1, 0],
        [0, board_size - 1],
        [board_size - 1, board_size - 1]
    ], dtype=np.float32)
    M = cv2.getPerspectiveTransform(corners_tl_tr_bl_br.astype(np.float32), dst)
    return M


def project_point(M: np.ndarray, x: float, y: float) -> Tuple[float, float]:
    """Projects an image point (x,y) with homography M to board plane coordinates (u,v)."""
    uv = cv2.perspectiveTransform(np.array([[[x, y]]], dtype=np.float32), M)[0, 0]
    return float(uv[0]), float(uv[1])

def _reconstruct_image_corners_from_homography(M_img_to_board: np.ndarray, board_size: int) -> np.ndarray:
    """
    Given image->board homography, reconstruct image-space board corners by applying inv(M)
    to the destination square corners. Returns TL, TR, BL, BR as (4,2) float32.
    """
    invM = np.linalg.inv(M_img_to_board)
    dst = np.array([
        [0, 0],
        [board_size - 1, 0],
        [0, board_size - 1],
        [board_size - 1, board_size - 1]
    ], dtype=np.float64)
    # Homogeneous transform of 4 points
    pts_h = np.hstack([dst, np.ones((4, 1))]) @ invM.T
    pts = (pts_h[:, :2] / pts_h[:, 2:3]).astype(np.float32)
    # Order TL, TR, BL, BR
    return order_corners_tl_tr_bl_br_np(pts)

def order_corners_tl_tr_bl_br_np(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points (unordered) as TL, TR, BL, BR using sums/diffs heuristic."""
    assert pts.shape == (4, 2)
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, bl, br], axis=0).astype(np.float32)

def _edge_lengths(corners_tl_tr_bl_br: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns lengths of top, bottom, left, right edges in image pixels."""
    TL, TR, BL, BR = corners_tl_tr_bl_br
    def L(a, b): return float(np.linalg.norm(a - b))
    top = L(TL, TR)
    bottom = L(BL, BR)
    left = L(TL, BL)
    right = L(TR, BR)
    return top, bottom, left, right

def estimate_tilt_strength_from_homography(M_img_to_board: np.ndarray, board_size: int) -> Tuple[float, float, float]:
    """
    Estimates perspective tilt strengths along ranks and files using edge foreshortening.
    Returns (s_rank, s_file, s_dom) in [0,1], where s_dom = max(s_rank, s_file).
    """
    corners = _reconstruct_image_corners_from_homography(M_img_to_board, board_size)  # TL,TR,BL,BR
    top, bottom, left, right = _edge_lengths(corners)

    # Avoid division by zero, clamp tiny lengths
    eps = 1e-6
    top = max(top, eps); bottom = max(bottom, eps)
    left = max(left, eps); right = max(right, eps)

    r_rank = top / bottom         # 1.0 means no rank foreshortening
    r_file = left / right         # 1.0 means no file foreshortening

    # Symmetric strength: |1 - ratio| clamped to [0, 1]
    s_rank = float(min(1.0, abs(1.0 - r_rank)))
    s_file = float(min(1.0, abs(1.0 - r_file)))
    s_dom = max(s_rank, s_file)
    return s_rank, s_file, s_dom
