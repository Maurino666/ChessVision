from __future__ import annotations

import datetime as _dt
from pathlib import Path

from board_recognition import train

# ---------------------------------------------------------------------------
# Path resolution helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the root of the repository.

    * Script mode  → folder containing *this* file (via ``__file__``).
    * Notebook     → one level above the current working directory.
    """
    try:  # running as a .py script
        return Path(__file__).resolve().parent.parent
    except NameError:  # likely inside Jupyter
        return Path.cwd().parent.resolve()


PROJECT_ROOT = _project_root()
DATASET_DIR  = PROJECT_ROOT / "datasets" / "synthetic_dataset"

TRAIN_ROOT = DATASET_DIR / "train" / "images"
TRAIN_CSV  = DATASET_DIR / "train" / "corners.csv"

TEST_ROOT  = DATASET_DIR / "test" / "images"
TEST_CSV   = DATASET_DIR / "test" / "corners.csv"

RUNS_BASE = PROJECT_ROOT / "runs" / "board_recognition"
RUNS_BASE.mkdir(parents=True, exist_ok=True)

# Hyper‑parameters (edit freely)
NUM_EPOCHS    = 1
BATCH_SIZE    = 6
NUM_WORKERS   = 8
LEARNING_RATE = 0.005
NUM_KEYPOINTS = 4
EVAL_BATCH_SIZE = 4

# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def main(
    *,
    num_epochs: int = NUM_EPOCHS,
    batch_size: int = BATCH_SIZE,
    num_workers: int = NUM_WORKERS,
    lr: float = LEARNING_RATE,
    num_keypoints: int = NUM_KEYPOINTS,
    eval_batch_size: int = EVAL_BATCH_SIZE,
):
    """Train on *train* split, then evaluate on *test* split in one go."""

    timestamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir   = RUNS_BASE / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)


    # print(f"trainroot: {TRAIN_ROOT},\ntraincsv: {TRAIN_CSV},\ntestroot: {TEST_ROOT},\ntestcsv: {TEST_CSV},\nrunsbase: {RUNS_BASE},\nrundir: {run_dir},\nckpt: {ckpt_path}")


    print("=== START TRAINING ===")
    train.train_model(
        dataset_root=str(DATASET_DIR),
        csv_name = "corners.csv",
        num_keypoints=num_keypoints,
        num_epochs=num_epochs,
        batch_size=batch_size,
        num_workers=num_workers,
        lr=lr,
        save_path=str(run_dir),
        eval_batch_size = eval_batch_size,
        debug_max_iterations = 20
    )

if __name__ == "__main__":
    main()
