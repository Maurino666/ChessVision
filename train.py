import torch
from torch.utils.data import DataLoader
import torchvision
from tqdm import tqdm

from dataset import ChessboardCornersDataset
from transforms import (
    get_train_augmentations,
    BatchKorniaTransformWrapper,
)

from torchvision.models.detection import keypointrcnn_resnet50_fpn

def get_keypoint_model(num_keypoints=4):
    model = keypointrcnn_resnet50_fpn(weights="COCO_V1")
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = \
        torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(
            in_channels=in_channels,
            num_keypoints=num_keypoints
        )
    return model

def batch_collate_fn(batch):
    """
    Minimal collate: just stack images [B,C,H,W] on CPU, and keep a list of targets.
    """
    images, targets = list(zip(*batch))
    batch_imgs = torch.stack(images, dim=0)  # CPU tensor
    return batch_imgs, list(targets)

def train_model(
    dataset_root: str,
    csv_path: str,
    num_keypoints: int = 4,
    num_epochs: int = 10,
    batch_size: int = 4,
    num_workers: int = 0,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    device: str = None,
    save_path: str = "keypoint_rcnn_chessboard.pth"
):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    print("Loading dataset (with CPU-side resize and coordinate scaling)...")
    dataset = ChessboardCornersDataset(
        root=dataset_root,
        csv_file=csv_path,
        resize_hw=(800, 800)
    )

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=batch_collate_fn  # purely CPU
    )

    print("Creating Keypoint R-CNN model...")
    model = get_keypoint_model(num_keypoints=num_keypoints)
    model.to(device)

    # We'll create a batch-based Kornia transform for GPU
    print("Creating Kornia batch transform wrapper...")
    kornia_transform = get_train_augmentations(p=0.5)
    batch_augmenter  = BatchKorniaTransformWrapper(kornia_transform)

    # Prepare optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for images, targets in loader_iter:
            # 'images_cpu' => [B,C,H,W] on CPU
            # 'targets_cpu' => list of dicts, each with CPU bounding boxes/keypoints

            # 1) Move them to GPU
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

        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {total_loss:.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")
