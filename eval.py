import torch
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from dataset import ChessboardCornersDataset
from train import get_keypoint_model, collate_fn


# from dataset import ChessboardCornersDataset
# from transforms import ...
# from train import get_keypoint_model, collate_fn

def evaluate_model(
    model_path: str,
    dataset_root: str,
    csv_path: str,
    num_keypoints: int = 4,
    device: str = None,
    max_images: int = 5
):
    """
    Loads a trained Keypoint R-CNN model from model_path, evaluates it on a small
    portion of the dataset (dataset_root, csv_path), and prints or shows results.

    Args:
        model_path: path to the .pth file with model weights
        dataset_root: directory with images
        csv_path: path to CSV
        num_keypoints: how many keypoints the model expects
        device: 'cuda' or 'cpu'
        max_images: how many images to run evaluation on and show predictions
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Create the dataset (without data augmentation, e.g. only a resize if needed)
    # dataset = ChessboardCornersDataset(root=dataset_root, csv_file=csv_path, transforms=test_transform)
    dataset = ChessboardCornersDataset(root=dataset_root, csv_file=csv_path, transforms=None)

    # Create model
    model = get_keypoint_model(num_keypoints=num_keypoints)
    print(f"Loading weights from {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Create a small DataLoader (batch_size=1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

    print("Starting evaluation...")
    images_shown = 0


    for images, targets in loader:
        images = [img.to(device) for img in images]
        # No targets needed for inference, unless we want to compute loss
        with torch.no_grad():
            predictions = model(images)

        # Convert image to PIL for visualization
        img_pil = F.to_pil_image(images[0].cpu())

        # Let's print or visualize the prediction for keypoints
        # predictions[0] should contain "boxes", "keypoints", "scores", ...
        pred = predictions[0]
        # E.g. "boxes", "keypoints", "scores", "keypoints_scores"

        print(f"Image {images_shown}:")
        print("Predicted boxes:", pred["boxes"])
        print("Predicted keypoints:", pred["keypoints"])
        print("Scores:", pred["scores"])

        # If you want to show the image + predicted keypoints:
        if images_shown < max_images:
            plt.figure()
            plt.imshow(img_pil)
            # draw keypoints
            kps = pred["keypoints"][0].cpu().numpy()  # If the model returns shape [N, K, 2]; check if you have multiple instances
            for (x, y, v) in kps:
                plt.scatter(x, y, c='red')
            plt.title(f"Prediction for image {images_shown}")
            plt.show()

        images_shown += 1
        if images_shown >= max_images:
            break

    print("Evaluation complete.")

def main():
    model_path = "keypoint_rcnn_chessboard.pth"
    dataset_root = "dataset/images"
    csv_path = "dataset/corners.csv"
    evaluate_model(model_path, dataset_root, csv_path)

if __name__ == "__main__":
    main()
