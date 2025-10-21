# board_state_builder/builder.py
# Purpose: Reconstruct a full chessboard state from raw model outputs.
# This module performs geometry (homography), assignment, orientation
# normalization, and FEN building, returning a structured result that
# the API can serialize as-is.

from __future__ import annotations
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

# Internal imports from the same package
from .geometry import get_homography_from_corners, order_corners_tl_tr_bl_br_np
from .assignment import assign_by_homography, assign_by_naive_full_image
from .orientation import rotate_grid_180, rotate_grid_90_cw, rotate_grid_90_ccw, rotate_conf
from .fen import build_fen_from_grid, algebraic_square_from_row_col


def _valid_corners(corners: Optional[np.ndarray]) -> bool:
    """Returns True if corners is a (4,2) finite float-like array."""
    if corners is None:
        return False
    arr = np.asarray(corners)
    return arr.shape == (4, 2) and np.isfinite(arr).all()


def _board_bbox_from_corners(corners: np.ndarray) -> List[float]:
    """Computes [x1, y1, x2, y2] bounding box from 4 corner points (image space)."""
    x1 = float(np.min(corners[:, 0])); y1 = float(np.min(corners[:, 1]))
    x2 = float(np.max(corners[:, 0])); y2 = float(np.max(corners[:, 1]))
    return [x1, y1, x2, y2]


def _apply_orientation(
    grid: List[List[Optional[str]]],
    conf: np.ndarray,
    white_side: str
) -> Tuple[List[List[Optional[str]]], np.ndarray]:
    """
    Rotates grid and confidence map so that white is at the bottom, given the
    board orientation in the input image (white_side).
    """
    side = (white_side or "").strip().lower()
    if side == "south":
        return grid, conf
    if side == "north":
        return rotate_grid_180(grid), rotate_conf(conf, "north")
    if side == "west":
        return rotate_grid_90_ccw(grid), rotate_conf(conf, "west")
    if side == "east":
        return rotate_grid_90_cw(grid), rotate_conf(conf, "east")
    # Unknown orientation: return unchanged
    return grid, conf


def _collect_pieces(
    grid_oriented: List[List[Optional[str]]],
    conf_oriented: np.ndarray
) -> List[Dict[str, Any]]:
    """
    Builds a flat list of pieces with algebraic coordinates and confidences
    from an oriented 8x8 grid and its confidence matrix.
    """
    out: List[Dict[str, Any]] = []
    for r in range(8):
        for c in range(8):
            ch = grid_oriented[r][c]
            if ch is None:
                continue
            sq = algebraic_square_from_row_col(r, c)
            conf_val = float(0.0 if conf_oriented[r, c] < 0 else conf_oriented[r, c])
            out.append({"square": sq, "piece": ch, "confidence": conf_val})
    return out


def build_board_state(
    *,
    image_shape: Tuple[int, int],
    detections: List[Tuple[np.ndarray, str, float]],   # (xyxy, class_name, conf) on ORIGINAL image
    corners: Optional[np.ndarray],                     # (4,2) TL,TR,BL,BR in ORIGINAL image pixels, or None
    white_side: str,                                   # "south" | "north" | "west" | "east"
    class_to_fen: Dict[str, str],
    board_size: int = 800,
    min_box_area: float = 80.0,
    board_score: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Reconstructs the chessboard state from raw detector outputs and optional board corners.

    Parameters
    ----------
    image_shape : (H, W) of the ORIGINAL image in pixels.
    detections  : list of detections as (xyxy, class_name, confidence).
    corners     : optional (4,2) array of board corners (TL, TR, BL, BR) in image pixels.
    white_side  : board orientation in the input image.
    class_to_fen: mapping from detector class names to FEN characters.
    board_size  : side length of the virtual rectified board (pixels) used for homography target.
    min_box_area: minimum detection area (in px^2) to consider during assignment.
    board_score : optional score reported by the board keypoint model.

    Returns
    -------
    Structured dict with:
      - "fen": str
      - "grid": 8x8 list of FEN chars or None
      - "pieces": list of {square, piece, confidence}
      - "board": {
          "found": bool,
          "white_side": str,
          "corners": [[x,y], ...] | None,
          "bbox": [x1,y1,x2,y2] | None,
          "homography": [[...],[...],[...]] | None,
          "score": float | None,
          "assignment_mode": "homography" | "naive_full_image"
        }
      - "diagnostics": {"num_detections_raw": int, "image_shape": [H, W]}
    """
    H, W = int(image_shape[0]), int(image_shape[1])

    # Choose geometry-aware path if valid corners are provided; otherwise use fallback.
    if _valid_corners(corners):
        corners_np = order_corners_tl_tr_bl_br_np(np.asarray(corners, dtype=np.float32))
        M = get_homography_from_corners(corners_np, board_size)
        grid_img, best_conf = assign_by_homography(
            detections=detections,
            class_to_fen=class_to_fen,
            M_img_to_board=M,
            board_size=board_size,
            min_box_area=float(min_box_area),
        )
        assignment_mode = "homography"
        board_found = True
        bbox = _board_bbox_from_corners(corners_np)
        homography_list = M.astype(float).tolist()
        corners_list = [[float(c[0]), float(c[1])] for c in corners_np]
        score_val = float(board_score) if board_score is not None else None
    else:
        grid_img, best_conf = assign_by_naive_full_image(
            detections=detections,
            class_to_fen=class_to_fen,
            image_w=W,
            image_h=H,
            min_box_area=float(min_box_area),
        )
        assignment_mode = "naive_full_image"
        board_found = False
        bbox = None
        homography_list = None
        corners_list = None
        score_val = None

    # Normalize orientation (white at the bottom)
    grid_fen, conf_fen = _apply_orientation(grid_img, best_conf, white_side)

    # Build FEN and piece list
    fen = build_fen_from_grid(grid_fen)
    pieces = _collect_pieces(grid_fen, conf_fen)

    # Assemble response-like structure
    result: Dict[str, Any] = {
        "fen": fen,
        "grid": grid_fen,
        "pieces": pieces,
        "board": {
            "found": board_found,
            "white_side": white_side,
            "corners": corners_list,
            "bbox": bbox,
            "homography": homography_list,
            "score": score_val,
            "assignment_mode": assignment_mode,
        },
        "diagnostics": {
            "num_detections_raw": int(len(detections)),
            "image_shape": [H, W],
        },
    }
    return result
