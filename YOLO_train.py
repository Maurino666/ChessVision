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
    def dual_real_val(trainer):
        # locate weights/last.pt saved at end of this epoch
        ckpt_path = Path(trainer.save_dir) / "weights" / "last.pt"

        # spin up a *new* YOLO object, which builds its own loss correctly
        eval_net = YOLO(str(ckpt_path))

        # run validation; all plots/metrics stored in same run folder
        res = eval_net.val(
            data="datasets/real_val.yaml",
            batch=trainer.args.batch,
            imgsz=trainer.args.imgsz,
            project=str(trainer.save_dir),
            name="real_val",
            exist_ok=True,
            verbose=False
        )

        # ④ log the numbers into the training metrics dict
        trainer.metrics["real_map50"] = round(res.box.map50, 4)
        trainer.metrics["real_map5095"] = round(res.box.map, 4)

    # register the hook (old-API in v8.3)
    yolo.add_callback("on_fit_epoch_end", dual_real_val)

    # ------------------------------------------------------------------
    # Launch training; workers=0 keeps the smoke-test predictable
    yolo.train(
        data=MIXED_YAML,
        epochs=EPOCHS,
        batch=4,
        imgsz=IMG_SIZE,
        close_mosaic=0,
        project="runs/detect",
        name="dual_val_test",  # name of the run folder
        exist_ok=False,          # let YOLO create dual_val_demo exp, exp2…
    )

    print("Completed. Mixed metrics + real-only plots live in runs/detect/dual_val_demo/")

if __name__ == "__main__":
    main()
