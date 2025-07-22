# train_chess_aug.py
# ------------------------------------------------------------
# Ultralytics-YOLO training with *every* official augmentation
# flag set directly in the .train() call.
# ------------------------------------------------------------
from ultralytics import YOLO

def main() -> None:
    model = YOLO("yolov10s.pt")                         # COCO-pretrained backbone

    model.train(
        data="datasets/synthetic.yaml",                 # your dataset
        epochs=150,
        imgsz=640,
        batch=-1,

        # ----------   COLOUR JITTER   ------------------------
        hsv_h=0.05,     # hue   ±5 %
        hsv_s=0.80,     # sat.  ±80 %
        hsv_v=0.50,     # value ±50 %

        # ----------   GEOMETRIC JITTER -----------------------
        degrees=30,      # small tilt
        translate=0.20, # ±20 % shift
        scale=0.70,     # zoom range [0.7, 1/0.7]
        shear=0.0,
        perspective=0.0007,

        # ----------   FLIPS ----------------------------------
        fliplr=0.5,     # horizontal flip
        flipud=0.0,     # NO vertical flip (keeps board legal)

        # ----------   MIXED-IMAGE AUGS -----------------------
        mosaic=1.0,          # on by default
        copy_paste=0.20,     # 20 % chance inside mosaic
        mixup=0.20,          # 20 % chance to blend two mosaics
        close_mosaic=20,     # disable mosaic for final 20 epochs

        # ----------   MISC -----------------------------------
        cache=False,         # keep RAM usage predictable
        device="cuda:0",     # change to "cpu" if no GPU
        name="chess_full_aug"
    )

if __name__ == "__main__":
    main()
