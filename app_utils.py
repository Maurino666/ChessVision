# app_utils.py
# Utilities for chessboard inference:
# - Image decode and small helpers
# - Board corner prediction (your torch model)
# - Homography building and point projection
# - Detection assignment to 8x8 cells (homography or naive)
# - Simple Ultralytics YOLO wrapper for piece detection
#
# Geometry-aware update:
# - When a homography is available, detections are anchored using a tilt-adaptive
#   bottom-band centroid before projection. This reduces "shifted upward" errors
#   under perspective, especially for tall pieces.

from typing import Optional, Tuple, List, Dict
import io
import numpy as np
import cv2
from PIL import Image
import torch

# Your module functions for the board model
from board_recognition import predict_on_image as board_predict


# -----------------------
# Image and basic helpers
# -----------------------

def decode_image_to_bgr(file_bytes: bytes) -> Optional[np.ndarray]:
    """Decodes JPEG/PNG bytes into a BGR numpy array; returns None on failure."""
    try:
        pil = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        return bgr
    except Exception:
        return None


def build_fen_from_grid(grid: List[List[Optional[str]]]) -> str:
    """
    Builds a full FEN string from an oriented 8x8 grid.
    Assumes grid[0] is rank 8, grid[7] is rank 1, and entries are FEN chars or None.
    """
    rows = []
    for r in range(8):
        empty = 0
        out = ""
        for c in range(8):
            ch = grid[r][c]
            if ch is None:
                empty += 1
            else:
                if empty > 0:
                    out += str(empty)
                    empty = 0
                out += ch
        if empty > 0:
            out += str(empty)
        rows.append(out)
    board = "/".join(rows)
    return f"{board} w - - 0 1"


def algebraic_square_from_row_col(row: int, col: int) -> str:
    """Converts oriented grid indices to algebraic notation. row 0 = rank 8, col 0 = file 'a'."""
    files = "abcdefgh"
    file_ch = files[col]
    rank_ch = str(8 - row)
    return f"{file_ch}{rank_ch}"


def rotate_grid_180(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 180 degrees."""
    return [list(reversed(row)) for row in reversed(grid)]


def rotate_grid_90_cw(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 90 degrees clockwise."""
    size = 8
    return [[grid[size - 1 - r2][c2] for r2 in range(size)] for c2 in range(size)]


def rotate_grid_90_ccw(grid: List[List[Optional[str]]]) -> List[List[Optional[str]]]:
    """Returns a new grid rotated by 90 degrees counter-clockwise."""
    size = 8
    return [[grid[r2][size - 1 - c2] for r2 in range(size)] for c2 in range(size)]


def rotate_conf(conf: np.ndarray, white_side: str) -> np.ndarray:
    """Rotates the 8x8 confidence array to match the oriented grid."""
    if white_side == "south":
        return conf
    if white_side == "north":
        return np.rot90(conf, 2)
    if white_side == "west":
        return np.rot90(conf, 1)   # CCW
    if white_side == "east":
        return np.rot90(conf, -1)  # CW
    raise ValueError("Invalid white_side")


def _order_corners_tl_tr_bl_br_np(pts: np.ndarray) -> np.ndarray:
    """Orders 4 points (unordered) as TL, TR, BL, BR using sums/diffs heuristic."""
    assert pts.shape == (4, 2)
    s = pts.sum(axis=1)
    diff = (pts[:, 0] - pts[:, 1])
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.stack([tl, tr, bl, br], axis=0).astype(np.float32)


def map_class_to_fen(cls_name: str, class_to_fen: Dict[str, str]) -> Optional[str]:
    """Returns FEN char for a detector class name with light normalization."""
    if cls_name is None:
        return None
    k = cls_name.strip()
    fen = class_to_fen.get(k)
    if fen is not None:
        return fen
    k_norm = k.replace("-", "_").lower()
    return class_to_fen.get(k_norm)


def _to_tensor_3hw_from_bgr(bgr: np.ndarray) -> torch.Tensor:
    """Converts BGR uint8 HxWx3 to float tensor [3,H,W] in RGB range [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t


# -----------------------
# Board-corner inference
# -----------------------

def predict_corners_with_board_model(board_model: torch.nn.Module, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Runs the keypoint model on the ORIGINAL image and returns:
      - corners: (4,2) TL,TR,BL,BR in image coords (float32)
      - score:   float score for the selected instance
    Returns (None, None) if no valid prediction.
    """
    image_tensor = _to_tensor_3hw_from_bgr(bgr)  # [3,H,W] float in [0,1]
    pred = board_predict(board_model, image_tensor, device=None)
    if pred is None or "keypoints" not in pred or "scores" not in pred:
        return None, None

    keypoints = pred["keypoints"]  # [N, K, 3]
    scores    = pred["scores"]     # [N]
    if keypoints is None or scores is None or len(scores) == 0 or keypoints.shape[0] == 0:
        return None, None

    top_idx = int(torch.as_tensor(scores).argmax().item())
    kps = keypoints[top_idx]  # [K,3]
    top_score = float(torch.as_tensor(scores)[top_idx].item())

    if kps.shape[0] < 4 or kps.shape[1] < 2:
        return None, None

    xy = kps[:, :2]
    if isinstance(xy, torch.Tensor):
        xy = xy.detach().cpu().numpy()
    xy = np.asarray(xy, dtype=np.float32)

    if xy.shape[0] != 4:
        xy = xy[:4, :]
    corners = _order_corners_tl_tr_bl_br_np(xy)
    return corners, top_score


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


# -----------------------
# Tilt-aware anchoring
# -----------------------

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
    return _order_corners_tl_tr_bl_br_np(pts)


def _edge_lengths(corners_tl_tr_bl_br: np.ndarray) -> Tuple[float, float, float, float]:
    """Returns lengths of top, bottom, left, right edges in image pixels."""
    TL, TR, BL, BR = corners_tl_tr_bl_br
    def L(a, b): return float(np.linalg.norm(a - b))
    top = L(TL, TR)
    bottom = L(BL, BR)
    left = L(TL, BL)
    right = L(TR, BR)
    return top, bottom, left, right


def _estimate_tilt_strength_from_homography(M_img_to_board: np.ndarray, board_size: int) -> Tuple[float, float, float]:
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


# -----------------------
# Assignment helpers
# -----------------------

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
    _, _, s_dom = _estimate_tilt_strength_from_homography(M_img_to_board, board_size)
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


# -----------------------
# Piece detector wrapper
# -----------------------

class SimpleDetector:
    """
    Ultralytics YOLO wrapper returning (xyxy, class_name, confidence) per detection.
    Keep this class small so it can be swapped if you change runtime.
    """
    def __init__(self, weights_path: str, conf_thres: float, iou_thres: float):
        from ultralytics import YOLO  # local import to avoid hard dependency on import time
        self.model = YOLO(weights_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.id_to_name = {int(k): v for k, v in self.model.names.items()}

    def detect_pieces(self, bgr: np.ndarray) -> List[Tuple[np.ndarray, str, float]]:
        """Runs detection on the ORIGINAL image and returns (xyxy, class_name, confidence)."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=rgb, conf=self.conf_thres, iou=self.iou_thres, verbose=False)
        out: List[Tuple[np.ndarray, str, float]] = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            cls  = r.boxes.cls.cpu().numpy().astype(int)
            for i in range(len(xyxy)):
                name = self.id_to_name.get(int(cls[i]), str(int(cls[i])))
                out.append((xyxy[i], name, float(conf[i])))
        return out
