# app.py
# FastAPI server for chess position extraction with:
# - YOLO runs on the ORIGINAL image (no warp before detection).
# - Board keypoint model only for geometry mapping of detection centers to squares.
# - Orientation taken ONLY from request: white_side ∈ {south, north, west, east}.
# - NEW: Response includes 'board' with position (corners, bbox, homography, score).
#
# Install:
#   pip install "fastapi>=0.111" "uvicorn[standard]>=0.30" numpy opencv-python pillow ultralytics torch torchvision
#
# Run:
#   uvicorn app:app --host 127.0.0.1 --port 8000 --reload

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import numpy as np
import cv2
from PIL import Image
import io
from typing import Dict, List, Optional, Tuple

import torch  # board model uses torch
from ultralytics import YOLO  # piece detector backend

# Your board module (must be importable)
from board_recognition import load_model as load_board_model
from board_recognition import predict_on_image as board_predict


# -----------------------
# Configuration
# -----------------------

PIECE_WEIGHTS_PATH = "runs/detect/full_aug_synthetic_3/weights/best.pt"
BOARD_MODEL_PATH   = "runs/board_recognition/20250907_231248/model.pth"

BOARD_SIZE = 800  # virtual square size for homography target (pixels)

# Detector class map (underscore keys as in your model)
CLASS_TO_FEN: Dict[str, str] = {
    "white_pawn":  "P",
    "black_pawn":  "p",
    "white_rook":  "R",
    "black_rook":  "r",
    "white_knight":"N",
    "black_knight":"n",
    "white_bishop":"B",
    "black_bishop":"b",
    "white_queen": "Q",
    "black_queen": "q",
    "white_king":  "K",
    "black_king":  "k",
}

DET_CONF_THRES = 0.35
DET_IOU_THRES  = 0.45
MIN_BOX_AREA   = 80  # px^2


# -----------------------
# Response schemas
# -----------------------

class PieceOut(BaseModel):
    square: str
    piece: str
    confidence: float

class BoardOut(BaseModel):
    found: bool
    white_side: str
    corners: Optional[List[List[float]]] = None      # TL,TR,BL,BR in original image pixels
    bbox: Optional[List[float]] = None               # [x1,y1,x2,y2] from corners
    homography: Optional[List[List[float]]] = None   # 3x3 image->board matrix
    score: Optional[float] = None                    # model's score for the selected board instance
    assignment_mode: str

class InferResponse(BaseModel):
    status: str
    fen: Optional[str] = None
    pieces: Optional[List[PieceOut]] = None
    board: Optional[BoardOut] = None
    diagnostics: Optional[dict] = None


# -----------------------
# Utilities
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
    """Builds a full FEN string from an oriented 8x8 grid."""
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

def map_class_to_fen(cls_name: str) -> Optional[str]:
    """Returns FEN char for a detector class name with light normalization."""
    if cls_name is None:
        return None
    k = cls_name.strip()
    fen = CLASS_TO_FEN.get(k)
    if fen is not None:
        return fen
    k_norm = k.replace("-", "_").lower()
    return CLASS_TO_FEN.get(k_norm)

def _to_tensor_3hw_from_bgr(bgr: np.ndarray) -> torch.Tensor:
    """Converts BGR uint8 HxWx3 to float tensor [3,H,W] in RGB range [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t


# -----------------------
# Models (load once)
# -----------------------

class SimpleDetector:
    """Ultralytics wrapper returning (xyxy, class_name, confidence) per detection."""
    def __init__(self, weights_path: str, conf_thres: float, iou_thres: float):
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

# Load models once
BOARD_MODEL = load_board_model(BOARD_MODEL_PATH, num_keypoints=4)
BOARD_DEVICE = next(BOARD_MODEL.parameters()).device
DETECTOR = SimpleDetector(PIECE_WEIGHTS_PATH, DET_CONF_THRES, DET_IOU_THRES)


# -----------------------
# Board corner prediction (returns corners and score)
# -----------------------

def predict_corners_with_board_model(bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Runs your keypoint model on the ORIGINAL image and returns:
      - corners: (4,2) TL,TR,BL,BR in image coords (float32)
      - score:   float score for the selected instance
    Returns (None, None) if no valid prediction.
    """
    image_tensor = _to_tensor_3hw_from_bgr(bgr)  # [3,H,W] float in [0,1]
    pred = board_predict(BOARD_MODEL, image_tensor, device=None)
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


# -----------------------
# FastAPI app
# -----------------------

app = FastAPI(title="ChessVision (YOLO on original + board position in response)", version="0.3.0")

@app.get("/healthz")
def healthz():
    return {"ok": True}

@app.post("/infer", response_model=InferResponse)
async def infer(
    file: UploadFile = File(...),
    white_side: str = Query(
        "south",
        description="Where the white side is in the INPUT image: south|north|west|east."
    )
):
    # Validate orientation
    white_side = white_side.strip().lower()
    if white_side not in {"south", "north", "west", "east"}:
        return JSONResponse(
            status_code=400,
            content={"status": "bad_request", "message": "white_side must be one of: south, north, west, east"}
        )

    # 1) Decode ORIGINAL image
    img_bytes = await file.read()
    bgr = decode_image_to_bgr(img_bytes)
    if bgr is None:
        return JSONResponse(status_code=400, content={"status": "bad_request", "message": "Cannot decode image."})
    H, W = bgr.shape[:2]

    # 2) Run YOLO on ORIGINAL image
    detections = DETECTOR.detect_pieces(bgr)

    # 3) Predict board corners on ORIGINAL image
    corners, board_score = predict_corners_with_board_model(bgr)
    have_corners = corners is not None and isinstance(corners, np.ndarray) and corners.shape == (4, 2)

    # Prepare grid (image-space assignment first)
    best_conf = np.full((8, 8), -1.0, dtype=float)
    grid_img: List[List[Optional[str]]] = [[None for _ in range(8)] for _ in range(8)]

    board_obj: BoardOut

    if have_corners:
        # Homography image->board
        dst = np.array([
            [0, 0],
            [BOARD_SIZE - 1, 0],
            [0, BOARD_SIZE - 1],
            [BOARD_SIZE - 1, BOARD_SIZE - 1]
        ], dtype=np.float32)
        M = cv2.getPerspectiveTransform(corners.astype(np.float32), dst)
        cell_size = BOARD_SIZE / 8.0

        # Assign each detection by projecting center to virtual board plane
        for xyxy, cls_name, conf in detections:
            x1, y1, x2, y2 = xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            fen_char = map_class_to_fen(cls_name)
            if fen_char is None:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            uv = cv2.perspectiveTransform(np.array([[[cx, cy]]], dtype=np.float32), M)[0, 0]
            u, v = float(uv[0]), float(uv[1])

            if not (0.0 <= u < BOARD_SIZE and 0.0 <= v < BOARD_SIZE):
                continue

            col = int(np.clip(u // cell_size, 0, 7))
            row = int(np.clip(v // cell_size, 0, 7))
            if conf > best_conf[row, col]:
                best_conf[row, col] = conf
                grid_img[row][col] = fen_char

        # Build board object
        x1 = float(np.min(corners[:, 0])); y1 = float(np.min(corners[:, 1]))
        x2 = float(np.max(corners[:, 0])); y2 = float(np.max(corners[:, 1]))
        board_obj = BoardOut(
            found=True,
            white_side=white_side,
            corners=[[float(c[0]), float(c[1])] for c in corners],
            bbox=[x1, y1, x2, y2],
            homography=M.astype(float).tolist(),
            score=board_score,
            assignment_mode="homography"
        )

    else:
        # Fallback: naive 8x8 over the WHOLE IMAGE (only OK if the board fills the frame)
        cell_w = W / 8.0
        cell_h = H / 8.0
        for xyxy, cls_name, conf in detections:
            x1, y1, x2, y2 = xyxy
            area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if area < MIN_BOX_AREA:
                continue
            fen_char = map_class_to_fen(cls_name)
            if fen_char is None:
                continue

            cx = 0.5 * (x1 + x2)
            cy = 0.5 * (y1 + y2)
            col = int(np.clip(cx // cell_w, 0, 7))
            row = int(np.clip(cy // cell_h, 0, 7))
            if conf > best_conf[row, col]:
                best_conf[row, col] = conf
                grid_img[row][col] = fen_char

        board_obj = BoardOut(
            found=False,
            white_side=white_side,
            corners=None,
            bbox=None,
            homography=None,
            score=None,
            assignment_mode="naive_full_image"
        )

    # 4) Rotate grid to STANDARD FEN orientation (white at bottom)
    if white_side == "south":
        grid_fen = grid_img
        conf_fen = best_conf
    elif white_side == "north":
        grid_fen = rotate_grid_180(grid_img)
        conf_fen = rotate_conf(best_conf, "north")
    elif white_side == "west":
        grid_fen = rotate_grid_90_ccw(grid_img)
        conf_fen = rotate_conf(best_conf, "west")
    else:  # east
        grid_fen = rotate_grid_90_cw(grid_img)
        conf_fen = rotate_conf(best_conf, "east")

    # 5) Build FEN and pieces[]
    fen = build_fen_from_grid(grid_fen)
    pieces_out: List[PieceOut] = []
    for r in range(8):
        for c in range(8):
            ch = grid_fen[r][c]
            if ch is None:
                continue
            sq = algebraic_square_from_row_col(r, c)
            conf_val = float(0.0 if conf_fen[r, c] < 0 else conf_fen[r, c])
            pieces_out.append(PieceOut(square=sq, piece=ch, confidence=conf_val))

    return InferResponse(
        status="ok",
        fen=fen,
        pieces=pieces_out,
        board=board_obj,
        diagnostics={
            "num_detections_raw": len(detections),
            "image_shape": [int(H), int(W)]
        }
    )


# Optional: run via `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
