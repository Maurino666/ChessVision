# app.py
# Thin FastAPI server: receives the image, runs models, delegates state reconstruction
# to board_state_builder.build_board_state, and returns a structured JSON response.

from __future__ import annotations
from typing import List, Optional, Dict

from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Model runtime and image decode utilities (unchanged locations)
from board_recognition import load_model as load_board_model
from app_utils import (
    decode_image_to_bgr,
    predict_corners_with_board_model,
    SimpleDetector,
)

# Board-state reconstruction façade
from state_builder import build_board_state

# -----------------------
# Configuration
# -----------------------

PIECE_WEIGHTS_PATH = "runs/detect/full_aug_synthetic_3/weights/best.pt"
BOARD_MODEL_PATH   = "runs/board_recognition/20250907_231248/model.pth"

BOARD_SIZE = 800  # virtual square size for homography target (pixels)

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
    corners: Optional[List[List[float]]] = None
    bbox: Optional[List[float]] = None
    homography: Optional[List[List[float]]] = None
    score: Optional[float] = None
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

app = FastAPI(title="ChessVision", version="0.4.0")

# Load models once at startup
BOARD_MODEL = load_board_model(BOARD_MODEL_PATH, num_keypoints=4)
BOARD_DEVICE = next(BOARD_MODEL.parameters()).device  # kept for introspection
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

    # 2) Run models on ORIGINAL image
    detections = DETECTOR.detect_pieces(bgr)  # [(xyxy, class_name, conf), ...]
    corners, board_score = predict_corners_with_board_model(BOARD_MODEL, bgr)  # (4,2) or None, score

    # 3) Delegate state reconstruction to the builder
    state = build_board_state(
        image_shape=(H, W),
        detections=detections,
        corners=corners,
        white_side=white_side,
        class_to_fen=CLASS_TO_FEN,
        board_size=BOARD_SIZE,
        min_box_area=float(MIN_BOX_AREA),
        board_score=board_score,
    )

    # 4) Marshal to the response model
    pieces_out: List[PieceOut] = [PieceOut(**p) for p in state.get("pieces", [])]
    board_out: Optional[BoardOut] = BoardOut(**state["board"]) if "board" in state else None

    return InferResponse(
        status="ok",
        fen=state.get("fen"),
        pieces=pieces_out,
        board=board_out,
        diagnostics=state.get("diagnostics", {}),
    )

# Optional: run via `python app.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
