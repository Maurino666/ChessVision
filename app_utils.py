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

from typing import Optional, Tuple, List
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


def _to_tensor_3hw_from_bgr(bgr: np.ndarray) -> torch.Tensor:
    """Converts BGR uint8 HxWx3 to float tensor [3,H,W] in RGB range [0,1]."""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return t

# Corner inference helper
def predict_corners_with_board_model(board_model: torch.nn.Module, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[float]]:
    """
    Runs the keypoint model on the ORIGINAL image and returns:
      - corners: (4,2) in image coords (float32), order may be arbitrary
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

    return xy.astype(np.float32), top_score


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
