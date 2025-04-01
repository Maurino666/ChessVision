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
                     If a tuple is provided, it is used directly (Kornia expects size in the format (H, W)).
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
    # boxes shape: [B, M, 4]
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
    # corners shape: [B, M, 4, 2]
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
    ensuring that images, keypoints, bounding boxes, and keypoint visibility (if present) are transformed consistently.

    It also adjusts keypoint visibility if any keypoints fall outside the final image boundaries after augmentation.

    Parameters:
        kornia_transform: An AugmentationSequential object (e.g., from get_train_augmentations),
                          configured with data_keys=["input", "keypoints", "bbox"] for bounding boxes.

    Processing Steps:
        1. Expand the image tensor from [C, H, W] to [1, C, H, W].
        2. Ensure keypoints are in shape [1, N, 2].
        3. If bounding boxes are present, ensure they are in shape [1, M, 4].
           Convert them to corners [1, M, 4, 2] so Kornia can treat them as polygons.
        4. Apply the Kornia transformation pipeline:
           - Pass the image, keypoints, and bounding box corners as positional arguments.
        5. Remove the batch dimension from the outputs:
           - Convert the transformed image from [1, C, H', W'] back to [C, H', W'].
           - Convert the transformed keypoints from [1, N, 2] back to [N, 2], then re-add a batch dimension to restore [1, N, 2].
           - Convert the transformed corners from [1, M, 4, 2] back to boxes [1, M, 4] if necessary.
        6. Update the target dictionary with the transformed keypoints and boxes.
        7. Adjust keypoint visibility based on whether the final keypoints lie within the image boundaries.

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
                - 'keypoints': Tensor of shape [1, K, 2].
                - 'keypoints_visible': (optional) Tensor of shape [1, K], where each value can be 0 (not visible), 1 (occluded), 2 (fully visible).
                - 'boxes': (optional) Tensor of shape [1, M, 4] in the format [xmin, ymin, xmax, ymax].
                - ... other annotations if needed.

        Returns:
            A tuple (aug_img, target) where:
                - aug_img is the transformed image tensor [C, H', W'].
                - target is the updated dictionary with transformed keypoints, boxes, and (if present) updated visibility.
        """
        # 1) Expand the image to batch dimension [1, C, H, W]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2) Ensure keypoints are [1, K, 2]
        kps = target["keypoints"]  # expected shape: [1, N, 2]
        if kps.dim() == 2:
            kps = kps.unsqueeze(0)
        elif kps.dim() != 3:
            raise ValueError("Keypoints tensor has unexpected dimensions. Expected [1, N, 2].")

        # Possibly store original visibility if present
        # shape [1, N]
        kps_vis = target.get("keypoints_visible", None)
        if kps_vis is not None:
            if kps_vis.dim() == 1:
                kps_vis = kps_vis.unsqueeze(0)
            elif kps_vis.dim() != 2:
                raise ValueError("keypoints_visible has unexpected dimensions. Expected [1, N].")

        # 3) Convert boxes to corners if present
        corners = None
        if "boxes" in target:
            boxes = target["boxes"]  # expected shape: [1, M, 4]
            if boxes.dim() == 2:
                boxes = boxes.unsqueeze(0)
            elif boxes.dim() != 3:
                raise ValueError("Boxes tensor has unexpected dimensions. Expected [1, M, 4].")
            corners = boxes_to_corners(boxes)  # [1, M, 4, 2]

        # 4) Apply the Kornia transformation pipeline
        if corners is not None:
            aug_img, aug_kps, aug_corners = self.kornia_transform(img_tensor, kps, corners)
        else:
            aug_img, aug_kps = self.kornia_transform(img_tensor, kps)

        # 5) Remove batch dimension and re-expand keypoints
        aug_img = aug_img.squeeze(0)        # [C, H', W']
        aug_kps = aug_kps.squeeze(0)        # [N, 2]
        aug_kps = aug_kps.unsqueeze(0)      # [1, N, 2]
        target["keypoints"] = aug_kps

        # If we have boxes, convert corners back to [xmin, ymin, xmax, ymax]
        if corners is not None:
            aug_corners = aug_corners.squeeze(0)   # [M, 4, 2]
            aug_corners = aug_corners.unsqueeze(0) # [1, M, 4, 2]
            aug_boxes = corners_to_boxes(aug_corners)  # [1, M, 4]
            target["boxes"] = aug_boxes

        # 6) If we have keypoints_visible, update it based on whether the keypoints remain in-bounds
        if kps_vis is not None:
            # shape [1, N]
            # We must check each keypoint's coordinate in [aug_kps], shape [1, N, 2]
            # and see if it lies within [0, W'] and [0, H'].
            aug_kps_2d = aug_kps[0]  # shape [N, 2]
            h_out, w_out = aug_img.shape[1], aug_img.shape[2]  # aug_img: [C, H', W']
            updated_vis = kps_vis.clone()  # shape [1, N]

            for i in range(aug_kps_2d.shape[0]):
                if updated_vis[0, i] > 0:  # e.g., 1 or 2 means previously visible
                    x_coord = aug_kps_2d[i, 0].item()
                    y_coord = aug_kps_2d[i, 1].item()
                    # Check bounds
                    if x_coord < 0 or x_coord >= w_out or y_coord < 0 or y_coord >= h_out:
                        # Out of the final image -> set visibility to 0
                        updated_vis[0, i] = 0
                    else:
                        # If originally 2 or 1, keep it. E.g. 2 remains 2, 1 remains 1
                        # (Or you can decide to unify them to 2)
                        pass

            target["keypoints_visible"] = updated_vis

        return aug_img, target
