# train_chess_aug.py
# ------------------------------------------------------------
# Ultralytics-YOLO training with *every* official augmentation
# flag set directly in the .train() call.
# ------------------------------------------------------------
from ultralytics import YOLO

def main() -> None:
    model = YOLO("runs/detect/full_aug_1024_high/weights/best.pt")
    model.train(
        data="datasets/combined_kq_mix.yaml",
        epochs=15,
        imgsz=1024,
        batch=4,
        patience=8,
        hsv_h=0.03, hsv_s=0.60, hsv_v=0.30,
        degrees=5, translate=0.10, scale=0.50, shear=0.0, perspective=0.0003,
        fliplr=0.5, flipud=0.0,
        mosaic=0.2, copy_paste=0.0, mixup=0.0, close_mosaic=10,
        lr0=0.001, lrf=0.1, momentum=0.937, weight_decay=0.0005,
        freeze=10,
        cls=1.5, box=7.5, dfl=1.0,
        device="cuda:0",
        project="runs/detect/full_aug_1024_high",
        name="finetune_kq_stage1_mix_frozen",
        verbose=True, plots=True, seed=42
    )


    model = YOLO("runs/detect/full_aug_1024_high/finetune_kq_stage1_mix_frozen/weights/best.pt")
    model.train(
            data="datasets/combined_kq_mix.yaml",
            epochs=8,
            imgsz=1024,
            batch=4,
            patience=6,
            hsv_h=0.03, hsv_s=0.60, hsv_v=0.30,
            degrees=5, translate=0.10, scale=0.50, shear=0.0, perspective=0.0003,
            fliplr=0.5, flipud=0.0,
            mosaic=0.2, copy_paste=0.0, mixup=0.0, close_mosaic=10,
            lr0=0.0005, lrf=0.1, momentum=0.937, weight_decay=0.0005,
            freeze=0,
            cls=1.5, box=7.5, dfl=1.0,
            device="cuda:0",
            project="runs/detect/full_aug_1024_high",
            name="finetune_kq_stage2_unfrozen",
            verbose=True, plots=True, seed=42
        )

    model = YOLO("runs/detect/full_aug_1024_high/finetune_kq_stage2_unfrozen/weights/best.pt")
    model.train(
        data="datasets/combined_kq_mix.yaml",
        epochs=5,
        imgsz=1280,
        batch=2,
        patience=4,
        hsv_h=0.02, hsv_s=0.50, hsv_v=0.25,
        degrees=3, translate=0.08, scale=0.45, shear=0.0, perspective=0.0002,
        fliplr=0.5, flipud=0.0,
        mosaic=0.0, copy_paste=0.0, mixup=0.0, close_mosaic=0,
        lr0=0.0003, lrf=0.1, momentum=0.937, weight_decay=0.0005,
        freeze=0,
        cls=1.5, box=7.5, dfl=1.0,
        device="cuda:0",
        project="runs/detect/full_aug_1024_high",
        name="finetune_kq_stage3_highres",
        verbose=True, plots=True, seed=42
    )



if __name__ == "__main__":
    main()


