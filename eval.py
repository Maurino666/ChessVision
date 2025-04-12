import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import ChessboardCornersDataset
from train import get_keypoint_model

def evaluate_model(
    model_path: str,
    dataset_root: str,
    csv_path: str,
    num_keypoints: int = 4,
    device: str = None,
    max_images: int = 5
):
    """
    Evaluates the trained Keypoint R-CNN on the resized dataset (no random augment).
    Shows or prints bounding boxes and keypoints for a few images.
    """
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Create dataset with the same resize, but no augmentations
    dataset = ChessboardCornersDataset(
        root=dataset_root,
        csv_file=csv_path,
        resize_hw=(800, 800)
    )

    # Create the model
    model = get_keypoint_model(num_keypoints=num_keypoints)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Evaluate with batch_size=1 for simplicity
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    print("Starting evaluation...")
    images_shown = 0

    for images, targets in loader:
        # 'images' is shape [1, 3, H, W] for batch_size=1
        # Move to GPU
        images = images.to(device)  # => [1, 3, 800, 800]

        # Convert [1, 3, H, W] -> list of length 1, each [3, H, W]
        images_list = list(images)  # => [ tensor([3, H, W]) ]

        with torch.no_grad():
            # Keypoint R-CNN expects a list of 3D tensors => pass images_list
            predictions = model(images_list)

        # Convert the first image back to PIL for visualization
        # images_list[0] is shape [3, H, W], on GPU => move to CPU
        img_pil = F.to_pil_image(images_list[0].cpu())
        pred = predictions[0]  # The model returns a list of predictions (one per image in the list)

        print(f"Image {images_shown}:")
        print("Predicted boxes:", pred.get("boxes", None))
        print("Predicted keypoints:", pred.get("keypoints", None))
        print("Scores:", pred.get("scores", None))

        if images_shown < max_images:
            plt.figure()
            plt.imshow(img_pil)

            # If there are predicted keypoints, plot them
            if "keypoints" in pred and len(pred["keypoints"]) > 0:
                # keypoints[0] if multiple instances
                # shape [num_kpts, 3]
                kps = pred["keypoints"][0].cpu().numpy()
                for (x, y, v) in kps:
                    plt.scatter(x, y, c='red')

            plt.title(f"Prediction for image {images_shown}")
            plt.show()

        images_shown += 1
        if images_shown >= max_images:
            break

    print("Evaluation complete.")
