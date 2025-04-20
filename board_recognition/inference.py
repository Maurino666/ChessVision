import torch
from torchvision.transforms.functional import to_pil_image
import matplotlib.pyplot as plt

from .model import get_keypoint_model


def load_model(model_path: str, num_keypoints: int = 4, device: str = None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = get_keypoint_model(num_keypoints=num_keypoints)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_on_image(model, image_tensor: torch.Tensor, device: str = None):
    """
    Args:
        model: a loaded Keypoint R-CNN model
        image_tensor: a torch.Tensor of shape [3, H, W]
        device: 'cuda' or 'cpu'

    Returns:
        pred: dictionary with 'keypoints', 'boxes', 'scores', etc.
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
        return predictions[0]


def visualize_prediction(image_tensor: torch.Tensor, prediction: dict, title: str = "Prediction"):
    """
    Show predicted keypoints on an image.

    Args:
        image_tensor: [3, H, W] tensor
        prediction: prediction dict from model
        title: title for the plot
    """
    img = to_pil_image(image_tensor.cpu())

    plt.figure()
    plt.imshow(img)

    if "keypoints" in prediction and len(prediction["keypoints"]) > 0:
        kps = prediction["keypoints"][0].cpu().numpy()  # shape [4, 3]
        for (x, y, v) in kps:
            if v > 0:
                plt.scatter(x, y, c='red', s=40)

    if "boxes" in prediction and len(prediction["boxes"]) > 0:
        box = prediction["boxes"][0].cpu().numpy()  # xmin, ymin, xmax, ymax
        x0, y0, x1, y1 = box
        plt.gca().add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                          edgecolor='lime', fill=False, linewidth=2))

    plt.title(title)
    plt.axis("off")
    plt.show()
