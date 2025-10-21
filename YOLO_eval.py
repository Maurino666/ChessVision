# eval_real_min.py
# Purpose: run validation on the real set and save metrics under runs/detect/chess_full_aug2/real_eval

from ultralytics import YOLO

def main():
    model = YOLO("runs/detect/full_aug_1024_high/finetune_kq_stage3_highres/weights/best.pt")  # your trained weights
    model.val(
        data="datasets/real2_val.yaml",                         # resolves via YOLO datasets_dir
        imgsz=1024,
        project="runs/detect/full_aug_1024_high",    # write inside your run
        name="real2_postfinetune_eval_full",                         # folder: runs/detect/chess_full_aug2/real_eval
        plots=True,
        save_txt=True,
        save_json=True,
        device="cuda:0"
    )

if __name__ == "__main__":
    main()