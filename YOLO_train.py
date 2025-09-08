# train_chess_aug.py
# ------------------------------------------------------------
# Ultralytics-YOLO training with *every* official augmentation
# flag set directly in the .train() call.
# ------------------------------------------------------------
from ultralytics import YOLO

def main() -> None:
    model = YOLO("yolov10s.pt")                         # COCO-pretrained backbone

    model.train(
        data="datasets/real2.yaml",                 # your dataset
        epochs=150,
        imgsz=800,
        batch=-1,
        patience=10,

        # ----------   COLOUR JITTER   ------------------------
        #hsv_h=0.05,     # hue   ±5 %
        #hsv_s=0.80,     # sat.  ±80 %
        #hsv_v=0.40,     # value ±50 %

        # ----------   GEOMETRIC JITTER -----------------------
        #degrees=30,      # small tilt
        #translate=0.20, # ±20 % shift
        #scale=0.70,     # zoom range [0.7, 1/0.7]
        #shear=0.0,
        #perspective=0.0007,

        # ----------   FLIPS ----------------------------------
        #fliplr=0.5,     # horizontal flip
        #flipud=0.0,     # NO vertical flip (keeps board legal)

        # ----------   MIXED-IMAGE AUGS -----------------------
        #mosaic=1.0,          # on by default
        #copy_paste=0.20,     # 20 % chance inside mosaic
        #mixup=0.20,          # 20 % chance to blend two mosaics
        #close_mosaic=20,     # disable mosaic for final 20 epochs

        # ----------   MISC -----------------------------------
        cache=False,         # keep RAM usage predictable
        device="cuda:0",     # change to "cpu" if no GPU
        name="real_train_2",

        verbose=True,       # print training progress
    )

    model = YOLO("yolov10s.pt")
    model.train(
        data="datasets/real.yaml",  # your dataset
        epochs=150,
        imgsz=800,
        batch=-1,
        patience=10,

        # ----------   COLOUR JITTER   ------------------------
        hsv_h=0.05,     # hue   ±5 %
        hsv_s=0.80,     # sat.  ±80 %
        hsv_v=0.40,     # value ±50 %

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
        cache=False,  # keep RAM usage predictable
        device="cuda:0",  # change to "cpu" if no GPU
        name="full_aug_real_train",

        verbose=True,  # print training progress
    )

    model = YOLO("yolov10s.pt")
    model.train(
        data="datasets/real2.yaml",  # your dataset
        epochs=150,
        imgsz=800,
        batch=-1,
        patience=10,

        # ----------   COLOUR JITTER   ------------------------
        hsv_h=0.05,  # hue   ±5 %
        hsv_s=0.80,  # sat.  ±80 %
        hsv_v=0.40,  # value ±50 %

        # ----------   GEOMETRIC JITTER -----------------------
        degrees=30,  # small tilt
        translate=0.20,  # ±20 % shift
        scale=0.70,  # zoom range [0.7, 1/0.7]
        shear=0.0,
        perspective=0.0007,

        # ----------   FLIPS ----------------------------------
        fliplr=0.5,  # horizontal flip
        flipud=0.0,  # NO vertical flip (keeps board legal)

        # ----------   MIXED-IMAGE AUGS -----------------------
        mosaic=1.0,  # on by default
        copy_paste=0.20,  # 20 % chance inside mosaic
        mixup=0.20,  # 20 % chance to blend two mosaics
        close_mosaic=20,  # disable mosaic for final 20 epochs

        # ----------   MISC -----------------------------------
        cache=False,  # keep RAM usage predictable
        device="cuda:0",  # change to "cpu" if no GPU
        name="full_aug_real_train_2",

        verbose=True,  # print training progress
    )



if __name__ == "__main__":
    main()
