import torch
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision
from dataset import ChessboardCornersDataset
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from tqdm import tqdm

# Importa la tua classe di Dataset
# from dataset import ChessboardCornersDataset
# Importa eventuali trasformazioni
# from transforms import get_train_augmentations, KorniaTransformWrapper
# Oppure sostituisci con i riferimenti corretti ai tuoi file

def get_keypoint_model(num_keypoints=4):
    """
    Creates a Keypoint R-CNN model pre-trained on COCO,
    then replaces the final keypoint head with one that predicts 'num_keypoints'.
    """
    model = keypointrcnn_resnet50_fpn(weights="COCO_V1")
    in_channels = model.roi_heads.keypoint_predictor.kps_score_lowres.in_channels
    model.roi_heads.keypoint_predictor = \
        torchvision.models.detection.keypoint_rcnn.KeypointRCNNPredictor(
            in_channels=in_channels,
            num_keypoints=num_keypoints
        )
    return model

def collate_fn(batch):
    """
    Custom collate function to group images and targets in lists,
    required by TorchVision detection models.
    """
    return list(zip(*batch))

def train_model(
    dataset_root: str,
    csv_path: str,
    num_keypoints: int = 4,
    num_epochs: int = 10,
    batch_size: int = 2,
    lr: float = 0.005,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    device: str = None,
    save_path: str = "keypoint_rcnn_chessboard.pth"
):
    """
    Trains a Keypoint R-CNN model on the dataset specified by (dataset_root, csv_path).

    Args:
        dataset_root: path to the directory with images
        csv_path: path to the CSV file with keypoints/boxes info
        num_keypoints: number of keypoints (e.g. 4 for corners)
        num_epochs: how many epochs to train
        batch_size: size of each batch
        lr, momentum, weight_decay: optimizer hyperparameters
        device: 'cuda' or 'cpu' (if None, auto-detect)
        save_path: where to save the final model
    """
    # Device setup
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Create the dataset
    # Example:
    # dataset = ChessboardCornersDataset(root=dataset_root, csv_file=csv_path, transforms=None)
    # Se hai un pipeline Kornia, potresti fare:
    # train_transforms = KorniaTransformWrapper(get_train_augmentations(800, p=0.5))
    # dataset = ChessboardCornersDataset(root=dataset_root, csv_file=csv_path, transforms=train_transforms)

    print("Loading dataset...")
    dataset = ChessboardCornersDataset(root=dataset_root, csv_file=csv_path, transforms=None)

    # DataLoader
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )

    # Create model
    model = get_keypoint_model(num_keypoints=num_keypoints)
    model.to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        # Wrappiamo il train_loader in tqdm per vedere la barra di avanzamento
        loader_iter = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}", leave=True)

        for images, targets in loader_iter:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if hasattr(v, "to") else v for k, v in t.items()} for t in targets]

            # Forward -> get losses
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_loss += losses.item()

            # Backprop
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Aggiorniamo la descrizione della progress bar con la loss corrente
            loader_iter.set_postfix({"loss": f"{losses.item():.4f}"})

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")

    print("Training complete. Saving model...")
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def main():
    # Esempio di parametri
    dataset_root = "dataset/images"
    csv_path = "dataset/corners.csv"
    train_model(dataset_root, csv_path)

if __name__ == "__main__":
    main()
