# train_dual_val.py   ─ place in ChessVision root
from ultralytics import YOLO
from pathlib import Path

MIXED_YAML = "datasets/chess.yaml"        # train + mixed-val
REAL_YAML  = "datasets/real_val.yaml"     # val split = real-only
IMG_SIZE   = 640
BATCH      = 4
EPOCHS     = 10          # put any number you want

def main():
    # ------------------------------------------------------------------
    # Build the high-level YOLO wrapper once
    yolo = YOLO("yolov10s.pt")                # COCO-pretrained backbone

    # ------------------------------------------------------------------
    # Callback: runs *after each epoch* and logs extra metrics/plots
    def dual_real_val(trainer, yolo_obj=yolo, real_yaml=REAL_YAML):
        """
        trainer  : DetectionTrainer instance created by model.train()
        yolo_obj : the high-level YOLO object that owns .val()
        real_yaml: data file whose `val:` (or `test:`) is the real-only set
        """
        res = yolo_obj.val(
            data=real_yaml,
            batch=trainer.args.batch,      # same batch size
            imgsz=trainer.args.imgsz,      # same image size
            plots=True,                    # save PR curves, confusion matrix…
            project=str(trainer.save_dir), # *same* run folder as training
            name="real_val",               # sub-folder inside save_dir
            exist_ok=True,                 # don’t create exp2, exp3…
            verbose=False,
        )
        # push the metrics into the epoch log so they appear in TensorBoard/CSV
        trainer.metrics["real_map50"]   = round(res.box.map50, 4)
        trainer.metrics["real_map5095"] = round(res.box.map,   4)

    # register the hook (old-API in v8.3)
    yolo.add_callback("on_fit_epoch_end", dual_real_val)

    # ------------------------------------------------------------------
    # Launch training; workers=0 keeps the smoke-test predictable
    yolo.train(
        data=MIXED_YAML,
        epochs=1,
        batch=4,
        imgsz=IMG_SIZE,
        project="runs/detect",
        name="dual_val_test",  # name of the run folder
        exist_ok=False,          # let YOLO create dual_val_demo exp, exp2…
    )

    print("✅ Done.  Mixed metrics + real-only plots live in runs/detect/dual_val_demo/")

if __name__ == "__main__":
    main()
