import torchvision.transforms.functional as F
import torch
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

class CustomResize:
    """
    A custom resize transform for PIL images along with associated bounding boxes and keypoints.

    This transform resizes the image and scales the keypoints and bounding boxes accordingly.

    Parameters:
        size (int or tuple): If an integer is provided, the image is resized to a square of (size, size).
                             If a tuple is provided, it should be in the format (new_width, new_height).

    Usage:
        transform = CustomResize(800)  # Resizes the image to 800x800.
    """
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, target):
        """
        Resizes the image and updates keypoints and bounding boxes.

        Parameters:
            img: PIL Image representing the input image.
            target: Dictionary containing:
                - 'keypoints': Tensor of shape (N, K, 2) where each keypoint is (x, y).
                - 'boxes': Tensor of shape (N, 4) in the format [xmin, ymin, xmax, ymax].
                - ... (other annotations)

        Processing Steps:
            1. Retrieve the original image dimensions.
            2. Resize the image to the new dimensions.
            3. Compute scaling factors for width and height.
            4. Scale keypoints using the computed factors.
            5. Scale bounding boxes using the computed factors.

        Returns:
            A tuple (img_resized, target) where:
                - img_resized is the resized PIL Image.
                - target is the updated dictionary with scaled keypoints and bounding boxes.
        """
        old_w, old_h = img.size
        new_w, new_h = self.size

        # 1) Resize the image
        img_resized = F.resize(img, (new_h, new_w))

        # 2) Compute the scaling factors
        scale_x = new_w / old_w
        scale_y = new_h / old_h

        # 3) Update keypoints (each keypoint is (x, y))
        keypoints = target['keypoints'].clone()
        keypoints[..., 0] *= scale_x
        keypoints[..., 1] *= scale_y
        target['keypoints'] = keypoints

        # 4) Update bounding boxes [xmin, ymin, xmax, ymax]
        boxes = target['boxes'].clone()
        boxes[..., [0, 2]] *= scale_x
        boxes[..., [1, 3]] *= scale_y
        target['boxes'] = boxes

        return img_resized, target


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
                     If a tuple is provided, it is used directly (expected format: (height, width) for Kornia).
        p (float): The probability for applying each random transformation.

    Returns:
        An AugmentationSequential object that applies the defined augmentations to both the image and keypoints.
    """
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    transform = AugmentationSequential(
        # 1) Resize to the fixed size (Kornia expects size in the format (H, W))
        K.Resize(size=resize_size),

        # 2) Random Horizontal Flip
        K.RandomHorizontalFlip(p=p),

        # 3) Random Rotation within ±15°
        K.RandomRotation(degrees=15.0, p=p),

        # 4) Color Jitter
        K.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
            p=p
        ),

        data_keys=["input", "keypoints"]
    )
    return transform


class KorniaTransformWrapper:
    """
    A wrapper for a Kornia AugmentationSequential pipeline that adapts it for single-image data.

    This wrapper handles conversion between single-image tensors and the batch format required by Kornia,
    ensuring that both images and keypoints are transformed consistently.

    Parameters:
        kornia_transform: An AugmentationSequential object (e.g., from get_train_augmentations).

    Processing Steps:
        1. Expand the image tensor from [C, H, W] to [1, C, H, W].
        2. Ensure keypoints are in the shape [1, N, 2].
        3. Apply the Kornia transformation pipeline.
        4. Remove the batch dimension from the outputs:
           - Convert the transformed image from [1, C, H', W'] back to [C, H', W'].
           - Convert the transformed keypoints from [1, N, 2] back to [N, 2], then re-add batch dimension to obtain [1, N, 2].
        5. Update the target dictionary with the transformed keypoints.

    Usage:
        transform_wrapper = KorniaTransformWrapper(get_train_augmentations(800, p=0.5))
    """
    def __init__(self, kornia_transform):
        self.kornia_transform = kornia_transform

    def __call__(self, img_tensor, target):
        """
        Parameters:
            img_tensor: Tensor of shape [C, H, W] representing the input image.
            target: Dictionary containing:
                - 'keypoints': Tensor of shape [1, K, 2] (where K is the number of keypoints)
                - 'boxes': Bounding box information (if available)
                - ... (other annotations)

        Processing Steps:
            1. Expand the image tensor to include a batch dimension:
               - Convert from [C, H, W] to [1, C, H, W].
            2. Ensure keypoints are in the correct batch format:
               - Adjust keypoints to shape [1, N, 2] if not already in that format.
            3. Apply the Kornia transformation pipeline:
               - Pass the image and keypoints to the Kornia transform.
            4. Remove the added batch dimension from the outputs:
               - Convert the transformed image from [1, C, H', W'] back to [C, H', W'].
               - Convert the transformed keypoints from [1, N, 2] back to [N, 2], then re-add batch dimension to restore [1, N, 2].
            5. Update the target dictionary:
               - Replace the original keypoints with the transformed keypoints (ensuring they have shape [1, N, 2]).

        Returns:
            A tuple (aug_img, target) where:
                - aug_img is the transformed image tensor with shape [C, H', W'].
                - target is the updated dictionary with transformed keypoints.
        """
        # Step 1: Expand image tensor to batch dimension
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]

        # Step 2: Ensure keypoints have shape [1, N, 2]
        kps = target["keypoints"]  # Expected shape: [1, N, 2]
        kps = kps.squeeze(0)       # -> [N, 2]
        kps = kps.unsqueeze(0)     # -> [1, N, 2]

        # Step 3: Apply the Kornia transformation
        aug_img, aug_kps = self.kornia_transform(img_tensor, keypoints=kps)

        # Step 4: Remove batch dimension from outputs
        aug_img = aug_img.squeeze(0)   # -> [C, H', W']
        aug_kps = aug_kps.squeeze(0)   # -> [N, 2]
        aug_kps = aug_kps.unsqueeze(0) # Ensure shape is [1, N, 2]

        # Step 5: Update the target dictionary with the new keypoints
        target["keypoints"] = aug_kps

        return aug_img, target
