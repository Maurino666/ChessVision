import numpy as np
from typing import List, Tuple, Dict, Optional

from .mapping import map_class_to_fen
from .geometry import project_point, estimate_tilt_strength_from_homography


def _choose_anchor_params_from_tilt(s_dom: float) -> Tuple[float, float, float]:
    """
    Maps tilt strength to bottom-band parameters.
    Returns (k, mu, min_band_px) where:
      - k is band height as a fraction of box height (default ~0.25)
      - mu is position within the band from the bottom edge (0=on bottom, 1=top of band)
      - min_band_px clamps the band to a minimum pixel height
    We push mu closer to the bottom as tilt increases.
    """
    # Defaults chosen to be simple and robust; tune on a small validation set if needed.
    k_base = 0.25                  # 25% of box height
    mu_min, mu_max = 0.35, 0.65    # 0.5 is exact band centroid
    gamma = 0.7                    # curvature for a smooth, monotonic mapping
    mu = mu_min + (mu_max - mu_min) * (s_dom ** gamma)
    min_band_px = 8.0              # keep a floor for tiny detections
    return k_base, mu, min_band_px


def _bottom_band_anchor(x1: float, y1: float, x2: float, y2: float, k: float, mu: float, min_band_px: float) -> Tuple[float, float]:
    """
    Computes a bottom-band anchor inside a detection box.
    - k: band height fraction of the box height
    - mu: position inside the band measured from the bottom (0=bottom edge, 1=top of band)
    - min_band_px: minimum absolute band height in pixels
    """
    h = max(0.0, y2 - y1)
    w = max(0.0, x2 - x1)
    if h <= 0.0 or w <= 0.0:
        # Degenerate box; fall back to bottom-center
        return float(0.5 * (x1 + x2)), float(y2)

    band_h = max(k * h, min_band_px)
    # Ensure band does not exceed the box height
    band_h = min(band_h, h)
    anchor_x = 0.5 * (x1 + x2)
    anchor_y = y2 - mu * band_h
    return float(anchor_x), float(anchor_y)

def assign_by_homography(
    detections: List[Tuple[np.ndarray, str, float]],
    class_to_fen: Dict[str, str],
    M_img_to_board: np.ndarray,
    board_size: int,
    min_box_area: float
) -> Tuple[List[List[Optional[str]]], np.ndarray]:
    """
    Assigns detections to 8x8 cells by projecting an anchor point via homography.
    This version uses a tilt-adaptive bottom-band anchor derived from the board geometry.
    Returns (grid, best_conf) where:
      - grid is 8x8 of FEN chars or None
      - best_conf is 8x8 numpy array of max confidences per cell
    """
    # Estimate tilt strength once from the homography, then choose anchor params.
    _, _, s_dom = estimate_tilt_strength_from_homography(M_img_to_board, board_size)
    k, mu, min_band_px = _choose_anchor_params_from_tilt(s_dom)

    cell = board_size / 8.0
    best_conf = np.full((8, 8), -1.0, dtype=float)
    grid = [[None for _ in range(8)] for _ in range(8)]

    for xyxy, cls_name, conf in detections:
        x1, y1, x2, y2 = xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_box_area):
            continue
        fen_char = map_class_to_fen(cls_name, class_to_fen)
        if fen_char is None:
            continue

        # Compute a geometry-aware anchor near the board plane
        ax, ay = _bottom_band_anchor(float(x1), float(y1), float(x2), float(y2), k=k, mu=mu, min_band_px=min_band_px)

        # Project anchor to board plane and assign cell
        u, v = project_point(M_img_to_board, ax, ay)
        if not (0.0 <= u < board_size and 0.0 <= v < board_size):
            continue

        col = int(np.clip(u // cell, 0, 7))
        row = int(np.clip(v // cell, 0, 7))
        if conf > best_conf[row, col]:
            best_conf[row, col] = conf
            grid[row][col] = fen_char

    return grid, best_conf


def assign_by_naive_full_image(
    detections: List[Tuple[np.ndarray, str, float]],
    class_to_fen: Dict[str, str],
    image_w: int,
    image_h: int,
    min_box_area: float
) -> Tuple[List[List[Optional[str]]], np.ndarray]:
    """
    Assigns detections to 8x8 cells using a naive full-image grid (fallback).
    Uses a fixed bottom-band anchor to be more robust than strict bottom-center/center.
    Works only if the board essentially fills the frame.
    """
    # Without homography we cannot estimate tilt; use conservative defaults.
    k, mu, min_band_px = 0.25, 0.50, 8.0

    cell_w = image_w / 8.0
    cell_h = image_h / 8.0
    best_conf = np.full((8, 8), -1.0, dtype=float)
    grid = [[None for _ in range(8)] for _ in range(8)]

    for xyxy, cls_name, conf in detections:
        x1, y1, x2, y2 = xyxy
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        if area < float(min_box_area):
            continue
        fen_char = map_class_to_fen(cls_name, class_to_fen)
        if fen_char is None:
            continue

        # Bottom-band anchor in image coordinates
        ax, ay = _bottom_band_anchor(float(x1), float(y1), float(x2), float(y2), k=k, mu=mu, min_band_px=min_band_px)

        # Naive per-image 8x8 grid assignment
        col = int(np.clip(ax // cell_w, 0, 7))
        row = int(np.clip(ay // cell_h, 0, 7))
        if conf > best_conf[row, col]:
            best_conf[row, col] = conf
            grid[row][col] = fen_char

    return grid, best_conf
