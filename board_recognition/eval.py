from __future__ import annotations
from pathlib import Path

from torchvision.models.detection import KeypointRCNN
from tqdm.auto import tqdm

import torch
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from .inference import load_model
from .dataset import ChessboardCornersDataset
from .utils import reorder_corners_row_column

from typing import Dict

def _pick_pred_index(pred: Dict[str, torch.Tensor]) -> int:
    """Pick predicted instance to evaluate: highest score if available, else 0."""
    scores = pred.get("scores", None)
    if scores is not None and scores.numel() > 0:
        return int(scores.argmax().item())
    return 0

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

# PCK: Percentage of Correct Keypoints
def pck(
        dists: torch.Tensor,
        *,
        thresh: tuple[float, ...] | None = None,   # px values
        alpha:  tuple[float, ...] | None = None,   # factors
        bbox_diag: float | None = None
    ) -> dict[str, float]:


    # Validate inputs
    if (thresh is None) == (alpha is None):
        raise ValueError("Provide thresh OR alpha, not both.")
    if alpha and bbox_diag is None:
        raise ValueError("Need bbox_diag for alpha mode.")

    out = {}

    # --- absolute-threshold branch -----------------------------------------
    if thresh is not None:
        for t in thresh:
            acc = (dists < t).float().mean().item()
            out[f"pck_{int(t)}px"] = acc
        return out

    # --- normalised-threshold branch ---------------------------------------
    for a in alpha:
        t_px = a * bbox_diag
        acc = (dists < t_px).float().mean().item()
        key = f"pck_norm_{a:0.2f}".replace(".", "")  # e.g. 0.05 -> 'pck_norm_005'
        out[key] = acc
    return out


# mAP update function for bounding boxes
def update_bbox_map(
        metric: MeanAveragePrecision,
        pred:   Dict[str, torch.Tensor],
        tgt:    Dict[str, torch.Tensor],
    ) -> None:

    gt_boxes = {
        "boxes":  tgt["boxes"].cpu(),
        "labels": tgt["labels"].cpu()
    }

    pred_boxes = {
        "boxes":  pred["boxes"].cpu(),
        "scores": pred["scores"].cpu(),
        # If your model does not output class labels, make them all 1
        "labels": pred.get("labels",
                   torch.ones_like(pred["scores"]).int()).cpu()
    }

    # torchmetrics expects *lists* of dicts, even for batch_size = 1
    metric.update([pred_boxes], [gt_boxes])


# ---------------------------------------------------------------------------
# Public APIs
# ---------------------------------------------------------------------------

from board_recognition.collate import batch_collate_fn  # imported once


def evaluate_model(
        *,
        model: KeypointRCNN,
        dataset_root: str | Path,
        csv_path: str | Path,
        device: str | None = None,
        # ── optional batch size
        batch_size: int = 4,
        # ── optional epoch number
        epoch: int | None = None
    ):

    # Load the dataset e data loader
    ds = ChessboardCornersDataset(root=dataset_root,
                                  csv_file=csv_path,
                                  resize_hw=(800, 800))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=batch_collate_fn)

    # Create a metric for mean Average Precision
    map_metric = MeanAveragePrecision(
        box_format="xyxy",  # xmin, ymin, xmax, ymax
        iou_type="bbox",  # bounding-box mAP
        class_metrics=True  # keep per-class / per-IoU PR tensors
    )

    # Partial sums dictionary
    keypt_sum = {}
    n_objects = 0

    for images, targets in tqdm(dl, desc="Evaluating", unit="batch"):

        # Ensure tensors go to device (your collate returns a list of tensors)
        images_on_dev = [im.to(device) for im in images]

        with torch.no_grad():
            preds = model(images_on_dev)

        for pred, tgt in zip(preds, targets):

            tgt = {k: v.to(device) for k, v in tgt.items()}

            # --- bbox mAP update (safe even with 0 predictions) ---
            update_bbox_map(map_metric, pred, tgt)

            # --- guards: skip KP eval if empty pred/gt ---
            if ("keypoints" not in pred) or (pred["keypoints"] is None) or (pred["keypoints"].numel() == 0):
                continue
            if ("keypoints" not in tgt) or (tgt["keypoints"] is None) or (tgt["keypoints"].numel() == 0):
                continue

            # Select predicted instance
            pred_idx = _pick_pred_index(pred)
            if pred_idx >= pred["keypoints"].shape[0]:
                continue  # defensive

            # Shapes: pred["keypoints"] -> [N, K, 3], tgt["keypoints"] -> [G, K, 3] or [K, 3]
            pred_xy_all = pred["keypoints"][pred_idx, :, :2]  # [K,2]
            gt_kps_all = tgt["keypoints"]
            gt_xy_all = gt_kps_all[0, :, :2] if gt_kps_all.ndim == 3 else gt_kps_all[:, :2]  # [K,2]
            gt_vis_all = gt_kps_all[0, :, 2] if gt_kps_all.ndim == 3 else gt_kps_all[:, 2]  # [K]

            # We evaluate the first 4 board corners; reorder to [TL, TR, BL, BR]
            if pred_xy_all.shape[0] < 4 or gt_xy_all.shape[0] < 4:
                continue

            gt_xy_4 = gt_xy_all[:4]
            gt_vis_4 = gt_vis_all[:4]
            pred_xy_4 = pred_xy_all[:4]

            # Reorder both sets using your row/column rule
            gt_xy_4, gt_vis_4 = reorder_corners_row_column(gt_xy_4, gt_vis_4)
            # For the prediction, we only need the coordinates; pass a dummy vis (same shape) or reuse gt_vis_4
            pred_dummy_vis = torch.ones_like(gt_vis_4)
            pred_xy_4, _ = reorder_corners_row_column(pred_xy_4, pred_dummy_vis)

            # Visibility mask from GT (evaluate only visible/labeled points)
            vis_mask = (gt_vis_4 > 0)
            if vis_mask.sum().item() == 0:
                continue

            # L2 distance on visible points
            diff = pred_xy_4[vis_mask] - gt_xy_4[vis_mask]  # [M,2]
            dists = torch.linalg.norm(diff, dim=-1)  # [M]

            # BBox diagonal for normalized PCK
            x_min, y_min, x_max, y_max = tgt["boxes"][0]
            diag = torch.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2).item()
            if diag <= 0:
                continue

            img_stats = {
                "l2_px": dists.mean().item(),
                **pck(dists, thresh=(3., 5.)),
                **pck(dists, alpha=(.05, .10), bbox_diag=diag),
            }

            for k, v in img_stats.items():
                keypt_sum[k] = keypt_sum.get(k, 0.0) + v

            n_objects += 1

    stats = {"mean_" + k: (v / n_objects if n_objects > 0 else float("nan"))
             for k, v in keypt_sum.items()}

    results = map_metric.compute()  # dict of tensors

    # 2) Pull out the two most common fields and convert to plain floats
    stats["bbox_map"] = results["map"].item()  # average over IoU 0.50-0.95
    stats["bbox_map_50"] = results["map_50"].item()  # AP at IoU 0.50 (a bit looser)

    # 3) (Optional) keep precision/recall tensors for the PR curve
    # stats["pr_precision"] = results["precision"]  # tensor [IoU, 101]
    # stats["pr_recall"] = results["recall"]  # same shape]

    if epoch is not None:
        stats["epoch"] = epoch

    return stats


# ---------------------------------------------------------------------------

import json

def evaluate_trained_model(
        *,
        model_path: str | Path,
        dataset_root: str | Path,
        csv_path: str | Path,
        num_keypoints: int = 4,
        device: str | None = None,
        # ── optional persistence
        output_dir: str | Path | None = None,
        filename: str = "metrics.json",
        # ── optional batch size
        batch_size: int = 4,
        # ── optional epoch number
        epoch: int | None = None
    ) -> Dict[str, float]:
    # Load the model
    model = load_model(model_path, num_keypoints=num_keypoints, device=device)

    # Evaluate the model
    stats = evaluate_model(
        model=model,
        dataset_root=dataset_root,
        csv_path=csv_path,
        device=device,
        batch_size=batch_size,
        epoch=epoch
    )

    if output_dir is not None and output_dir != "":
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        serializable = {
            k: (v.item() if torch.is_tensor(v) else v)
            for k, v in stats.items()
        }

        out_path = output_dir / filename
        with out_path.open("w") as fp:
            json.dump(serializable, fp, indent=2)

        print(f"[Eval-Save] Metrics written to {out_path}")

    return stats

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

# ─────────────────────────────────────────────────────────────
# Interactive preview helper
# ─────────────────────────────────────────────────────────────
def preview_predictions(
    *,
    model: KeypointRCNN,
    dataset_root: str | Path,
    csv_path: str | Path,
    device: str | None = None,
    batch_size: int = 4,
    max_examples: int = 6,
):
    """
    Show GT (green) vs prediction (red) for the first *max_examples* images.

    Nothing is saved; useful for sanity-checking scale or obvious errors.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = ChessboardCornersDataset(root=dataset_root,
                                  csv_file=csv_path,
                                  resize_hw=(800, 800))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False,
                    collate_fn=batch_collate_fn)

    shown = 0
    plt.figure(figsize=(6, 6))

    for images, targets in dl:
        with torch.no_grad():
            preds = model(list(images.to(device)))

        for img_t, pred, tgt in zip(images, preds, targets):
            # ── plot
            ax = plt.gca()
            ax.imshow(F.to_pil_image(img_t))     # channel order OK
            ax.axis("off")

            # GT in green
            gt_kps = tgt["keypoints"][0, :, :2]
            ax.scatter(gt_kps[:, 0], gt_kps[:, 1],
                       s=30, c="lime", marker="o", label="GT")

            # Pred in red
            if pred["keypoints"].numel():
                pred_kps = pred["keypoints"][0, :, :2].cpu()
                ax.scatter(pred_kps[:, 0], pred_kps[:, 1],
                           s=30, c="red", marker="x", label="Pred")

            # title / scores
            if "scores" in pred:
                score = float(pred["scores"][0])
                ax.set_title(f"score={score:.2f}")

            plt.show()
            shown += 1
            if shown >= max_examples:
                return





