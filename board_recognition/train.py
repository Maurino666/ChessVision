from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .collate import batch_collate_fn
from .dataset import ChessboardCornersDataset
from .eval import evaluate_model
from .model import get_keypoint_model
from .plots import write_metric_plots
from .transforms import (
    get_train_augmentations,
    BatchKorniaTransformWrapper,
)


def train_model(
    dataset_root: str,
    csv_name: str = "corners.csv",
    num_keypoints: int = 4,
    num_epochs: int = 10,
    batch_size: int = 4,
    num_workers: int = 0,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    device: str = None,
    save_path: str = "run/board_recognition/train",
    eval_n = 1, # Evaluate every N epochs
    eval_batch_size: int | None = None,
    debug_max_iterations: int or None = None
):
    # device setup
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    if eval_batch_size is None:
        eval_batch_size = batch_size

    train_root = Path(dataset_root) / "train"
    train_csv = train_root / csv_name

    test_root = Path(dataset_root) / "test"
    test_csv = test_root / csv_name

    # dataset and dataloader setup
    print("Loading dataset (with CPU-side resize and coordinate scaling)...")
    dataset = ChessboardCornersDataset(
        root=str(train_root / "images"),
        csv_file=str(train_csv),
        resize_hw=(800, 800)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=batch_collate_fn  # purely CPU
    )

    # Model setup
    print("Creating Keypoint R-CNN model...")
    model = get_keypoint_model(num_keypoints=num_keypoints)
    model.to(device)

    # Batch-based Kornia transform for GPU
    print("Creating Kornia batch transform wrapper...")
    kornia_transform = get_train_augmentations(p=0.5)
    batch_augmenter  = BatchKorniaTransformWrapper(kornia_transform)

    # Optimizer setup
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Eval setup
    all_stats = []

    # Training loop
    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        seen = 0
        # tqdm progress bar for the current epoch
        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        # Epoch loop
        for images, targets in loader_iter:
            # 'images_cpu' => [B,C,H,W] on CPU
            # 'targets_cpu' => list of dicts, each with CPU bounding boxes/keypoints
            if debug_max_iterations is not None and seen > debug_max_iterations:
                break
            # 1) Move images and targets to the device (GPU or CPU)
            images = images.to(device)
            for t in targets:
                t["boxes"]     = t["boxes"].to(device)
                t["keypoints"] = t["keypoints"].to(device)
                t["labels"]    = t["labels"].to(device)

            # 2) Apply batch-based Kornia transform on GPU
            aug_imgs, aug_targets = batch_augmenter(images, targets)

            # 3) Pass to model
            # KeypointRCNN expects a list of images => [B, C, H',W']
            # We can pass list(aug_imgs) or just aug_imgs directly if the model is adapted.
            loss_dict = model(list(aug_imgs), aug_targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            loader_iter.set_postfix(loss=f"{losses.item():.4f}")
            seen += 1


        # End of epoch: evaluate the model on the test set
        if epoch % eval_n == 0 or epoch == num_epochs - 1:
            model.eval()
            all_stats.append(
                evaluate_model(
                    model=model,
                    dataset_root=test_root / "images",
                    csv_path=test_csv,
                    device=device,
                    batch_size = eval_batch_size,
                    epoch = epoch + 1
                )
            )
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    # Write metric plots to the output directory
    write_metric_plots(
        all_stats,
        output_dir=Path(save_path)
    )
    # Save the trained model
    print("Training complete. Saving model...")
    torch.save(model.state_dict(), save_path + "/model.pth")
    print(f"Model saved to {save_path}")
