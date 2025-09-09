# app.py
# Thin FastAPI server that wires configs, loads models, and calls utilities from app_utils.
# Behavior:
# - YOLO runs on the ORIGINAL image.
# - Board keypoint model is used only for geometry mapping (homography).
# - Orientation is taken ONLY from request: white_side ∈ {south, north, west, east}.
# - Response includes 'board' with corners, bbox, homography, and score.

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import numpy as np
import torch

# Your board loader
from board_recognition import load_model as load_board_model

# Utilities split out
from app_utils import (
    decode_image_to_bgr,
    build_fen_from_grid,
    algebraic_square_from_row_col,
    rotate_grid_180, rotate_grid_90_cw, rotate_grid_90_ccw, rotate_conf,
    predict_corners_with_board_model, get_homography_from_corners,
    assign_by_homography, assign_by_naive_full_image,
    SimpleDetector
)

# -----------------------
# Configuration (your paths unchanged)
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
# App setup and models
# -----------------------

app = FastAPI(title="ChessVision", version="0.3.1")

# Load models once
BOARD_MODEL = load_board_model(BOARD_MODEL_PATH, num_keypoints=4)
BOARD_DEVICE = next(BOARD_MODEL.parameters()).device  # not used directly, but useful to keep
DETECTOR = SimpleDetector(PIECE_WEIGHTS_PATH, DET_CONF_THRES, DET_IOU_THRES)

@app.get("/healthz")
def healthz():
    return {"ok": True}

# -----------------------
# Endpoint
# -----------------------

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
    corners, board_score = predict_corners_with_board_model(BOARD_MODEL, bgr)
    have_corners = corners is not None and isinstance(corners, np.ndarray) and corners.shape == (4, 2)

    # 4) Assign detections to cells
    if have_corners:
        M = get_homography_from_corners(corners, BOARD_SIZE)
        grid_img, best_conf = assign_by_homography(
            detections=detections,
            class_to_fen=CLASS_TO_FEN,
            M_img_to_board=M,
            board_size=BOARD_SIZE,
            min_box_area=MIN_BOX_AREA
        )

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
        grid_img, best_conf = assign_by_naive_full_image(
            detections=detections,
            class_to_fen=CLASS_TO_FEN,
            image_w=W,
            image_h=H,
            min_box_area=MIN_BOX_AREA
        )
        board_obj = BoardOut(
            found=False,
            white_side=white_side,
            corners=None,
            bbox=None,
            homography=None,
            score=None,
            assignment_mode="naive_full_image"
        )

    # 5) Rotate grid to standard FEN orientation (white at bottom)
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

    # 6) Build FEN and pieces[]
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
