import torchvision.transforms.functional as F
import torch
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

from typing import Dict, Any

def get_train_augmentations(resize_size=800, p=0.5):
    """
    Constructs an AugmentationSequential pipeline using Kornia for training augmentations.

    The pipeline includes:
        1. Resize: Resizes the image (and keypoints) to a fixed size.
        2. Random Horizontal Flip: Flips the image horizontally with probability p.
        3. Random Rotation: Rotates the image within ±15° with probability p.
        4. Color Jitter: Randomly alters brightness, contrast, saturation, and hue with probability p.

    Parameters:
        resize_size: Either an int or a tuple. If an int is provided, it is converted to (resize_size, resize_size).
                     If a tuple is provided, it is used directly (Kornia expects (H, W)).
        p (float): The probability for applying each random transformation.

    Returns:
        An AugmentationSequential object that applies the defined augmentations to the image, keypoints, and bbox.
    """
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    transform = AugmentationSequential(
        K.Resize(size=resize_size),
        K.RandomHorizontalFlip(p=p),
        K.RandomRotation(degrees=15.0, p=p),
        K.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=p
        ),
        data_keys=["input", "keypoints", "bbox"]
    )
    return transform

def get_test_augmentations(resize_size=800):
    """
    Constructs an AugmentationSequential pipeline using Kornia for test-time transformations.
    In test mode, we generally apply only deterministic transformations (e.g., a fixed resize)
    to standardize the image size.

    Parameters:
        resize_size: An int or a tuple. If int, it is converted to (resize_size, resize_size).
                     Otherwise, Kornia expects (H, W).
    Returns:
        An AugmentationSequential object that applies only a fixed resize (to the image, keypoints, and bbox).
    """
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    transform = AugmentationSequential(
        K.Resize(size=resize_size),
        data_keys=["input", "keypoints", "bbox"]
    )
    return transform

def boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Converts bounding boxes from shape [B, M, 4] (xmin, ymin, xmax, ymax) to
    corners shape [B, M, 4, 2].

    Each box is transformed into four vertices:
      [ [xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax] ]
    """
    xmin = boxes[..., 0]
    ymin = boxes[..., 1]
    xmax = boxes[..., 2]
    ymax = boxes[..., 3]

    top_left     = torch.stack([xmin, ymin], dim=-1)
    top_right    = torch.stack([xmax, ymin], dim=-1)
    bottom_right = torch.stack([xmax, ymax], dim=-1)
    bottom_left  = torch.stack([xmin, ymax], dim=-1)

    # corners shape: [B, M, 4, 2]
    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=-2)
    return corners

def corners_to_boxes(corners: torch.Tensor) -> torch.Tensor:
    """
    Converts corners from shape [B, M, 4, 2] back to
    axis-aligned bounding boxes [B, M, 4] (xmin, ymin, xmax, ymax).

    We assume that after transformation, each set of 4 corners still describes
    a quadrilateral. We compute:
        xmin = min(x_coords), xmax = max(x_coords),
        ymin = min(y_coords), ymax = max(y_coords)
    """
    x_coords = corners[..., 0]
    y_coords = corners[..., 1]

    xmin = x_coords.min(dim=-1)[0]
    xmax = x_coords.max(dim=-1)[0]
    ymin = y_coords.min(dim=-1)[0]
    ymax = y_coords.max(dim=-1)[0]

    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)  # [B, M, 4]
    return boxes

class KorniaTransformWrapper:
    """
    A wrapper for a Kornia AugmentationSequential pipeline that adapts it for single-image data.

    This wrapper handles conversion between single-image tensors and the batch format required by Kornia,
    ensuring that images, keypoints, bounding boxes, and the keypoint visibility (stored in the 3rd channel) are
    transformed consistently.

    It also adjusts visibility if any keypoints fall outside the final image boundaries after augmentation.

    Parameters:
        kornia_transform: An AugmentationSequential object (e.g., from get_train_augmentations),
                          configured with data_keys=["input", "keypoints", "bbox"] for bounding boxes.

    Processing Steps:
        1. Expand the image tensor from [C, H, W] to [1, C, H, W].
        2. We expect 'keypoints' to have shape [1, N, 3] => (x, y, visibility).
           We separate coords [1, N, 2] from vis [1, N] and pass only coords to Kornia.
        3. If bounding boxes are present, ensure they are shape [1, M, 4], and convert to corners [1, M, 4, 2].
        4. Apply the Kornia transformation pipeline on (img, keypoint_coords, corners).
        5. Remove the batch dimension from outputs:
           - The final image is [C, H', W'].
           - The final keypoint coords are [N, 2]. We re-add dimension => [1, N, 2].
           - The final corners => [1, M, 4, 2], then convert back to boxes => [1, M, 4].
        6. Combine coords + original visibility => [1, N, 3], then set to 0 if the point is out of the image.
        7. Store the updated keypoints in target["keypoints"].

    Usage:
        transform_wrapper = KorniaTransformWrapper(get_train_augmentations(800, p=0.5))
    """

    def __init__(self, kornia_transform: AugmentationSequential):
        self.kornia_transform = kornia_transform

    def __call__(self, img_tensor: torch.Tensor, target: Dict[str, Any]):
        """
        Parameters:
            img_tensor: A tensor of shape [C, H, W] representing the input image.
            target: A dictionary containing:
                - 'keypoints': shape [1, N, 3] => each row is (x, y, visibility).
                - 'boxes': (optional) shape [1, M, 4], format [xmin, ymin, xmax, ymax].
                - ... other annotations if needed.

        Returns:
            A tuple (aug_img, target) where:
                - aug_img is the transformed image tensor [C, H', W'].
                - target is the updated dictionary with transformed keypoints in [1, N, 3]
                  and bounding boxes in [1, M, 4] if present.
        """
        # 1) Expand image to batch dimension [1, C, H, W]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2) keypoints shape => [1, N, 3]. We'll separate coords [x,y] from visibility
        keypoints_3d = target["keypoints"]  # expected [1, N, 3]
        if keypoints_3d.dim() == 2:
            # if shape was [N, 3], add the batch dimension
            keypoints_3d = keypoints_3d.unsqueeze(0)
        elif keypoints_3d.dim() != 3:
            raise ValueError("keypoints must be [1, N, 3] or [N, 3].")

        # split => coords [1, N, 2], visibility [1, N]
        coords = keypoints_3d[..., :2]      # [1, N, 2]
        visibility = keypoints_3d[..., 2]   # [1, N]

        # 3) If bounding boxes are present, convert to corners
        corners = None
        if "boxes" in target:
            boxes = target["boxes"]  # e.g. shape [1, M, 4]
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
            elif boxes.dim() != 3:
                raise ValueError("Boxes tensor has unexpected dimensions. Expected [1, M, 4].")
            corners = boxes_to_corners(boxes)  # [1, M, 4, 2]

        # 4) Apply pipeline
        if corners is not None:
            aug_img, aug_coords, aug_corners = self.kornia_transform(img_tensor, coords, corners)
        else:
            aug_img, aug_coords = self.kornia_transform(img_tensor, coords)

        # 5) Remove batch dimension
        aug_img = aug_img.squeeze(0)           # [C, H', W']
        aug_coords = aug_coords.squeeze(0)     # [N, 2]
        aug_coords = aug_coords.unsqueeze(0)   # [1, N, 2]

        if corners is not None:
            aug_corners = aug_corners.squeeze(0)   # [M, 4, 2]
            aug_corners = aug_corners.unsqueeze(0) # [1, M, 4, 2]
            aug_boxes = corners_to_boxes(aug_corners)  # [1, M, 4]
            target["boxes"] = aug_boxes

        # 6) Re-combine (coords + visibility) => shape [1, N, 3]
        # and set visibility=0 if keypoint is out-of-bounds
        h_out, w_out = aug_img.shape[1], aug_img.shape[2]
        updated_coords = aug_coords[0]  # shape [N, 2]
        updated_vis = visibility.clone()  # shape [1, N]

        # Check bounds
        for i in range(updated_coords.shape[0]):
            if updated_vis[0, i] > 0:  # e.g. 1 or 2 => previously visible
                x_coord = updated_coords[i, 0].item()
                y_coord = updated_coords[i, 1].item()
                if x_coord < 0 or x_coord >= w_out or y_coord < 0 or y_coord >= h_out:
                    updated_vis[0, i] = 0  # out of the final image

        # build final shape [1, N, 3]
        updated_vis = updated_vis[0]  # => shape [N]
        final_keypoints = torch.cat(
            [updated_coords, updated_vis.unsqueeze(-1)],
            dim=-1
        )  # shape [N, 3]
        final_keypoints = final_keypoints.unsqueeze(0)  # shape [1, N, 3]

        target["keypoints"] = final_keypoints

        # 7) Return final results
        return aug_img, target
