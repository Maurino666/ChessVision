# Chess-Vision

**Chess-Vision** is a computer vision project designed to automatically recognize the state of a real chessboard from a single image.  
It combines deep learning and geometric processing to detect the board, identify each piece, and reconstruct the complete game position in FEN notation.

---

## Architecture Overview

The system is structured into three main modules:

- **Board Detection Module** — estimates the four corners of the chessboard using a Keypoint R-CNN with a ResNet backbone.  
- **Piece Detection Module** — identifies and classifies all chess pieces using YOLOv10-s.  
- **State Builder** — integrates geometric and semantic information to generate the final FEN representation.

All components are orchestrated by a lightweight FastAPI server that exposes a REST endpoint (`/infer`).

---

## Key Features

- End-to-end pipeline for chessboard state recognition  
- Modular and scalable architecture  
- Synthetic-to-real training for robust generalization  
- Output in FEN format for compatibility with chess engines  

---

## Technologies

- Python  
- PyTorch  
- FastAPI  
- OpenCV  
- NumPy, Pandas, Matplotlib  
- Ultralytics YOLOv10-s for object detection  
- ResNet for keypoint estimation  
- Unity for dataset generation  
- CVAT and Roboflow for annotation  

---

## Repository Structure

| Path | Description |
|------|--------------|
| `app.py` | Main FastAPI entry point |
| `app_utils.py` | Image and I/O utilities |
| `board_recognition/` | Board detection model |
| `board_recognition_run/` | Training and evaluation scripts |
| `state_builder/` | Reconstruction and FEN conversion logic |
| `YOLO_train.py` | YOLO training pipeline |
| `YOLO_eval.py` | YOLO evaluation script |

---

## Future Work

- Real-time video-based recognition  
- Sequential interpretation of entire matches  
- Enhanced discrimination of visually similar pieces  
- Integration with symbolic reasoning or chess engines  

---

**Author:** Maurilio La Rocca
