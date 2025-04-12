from typing import Dict, Any, List
import torch
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential

def get_train_augmentations(p=0.5):
    """
    Defines a Kornia AugmentationSequential pipeline (batch-based).
    - RandomHorizontalFlip
    - RandomRotation
    - ColorJitter

    We do NOT do another resize here, because the Dataset
    already resized the images to a fixed shape.
    """
    transform = AugmentationSequential(
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

def boxes_to_corners(boxes: torch.Tensor) -> torch.Tensor:
    """
    Convert bounding boxes [B, M, 4] => corners [B, M, 4, 2].
    Order of corners: top-left, top-right, bottom-right, bottom-left.
    """
    xmin = boxes[..., 0]
    ymin = boxes[..., 1]
    xmax = boxes[..., 2]
    ymax = boxes[..., 3]

    top_left     = torch.stack([xmin, ymin], dim=-1)
    top_right    = torch.stack([xmax, ymin], dim=-1)
    bottom_right = torch.stack([xmax, ymax], dim=-1)
    bottom_left  = torch.stack([xmin, ymax], dim=-1)

    corners = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=-2)
    return corners

def corners_to_boxes(corners: torch.Tensor) -> torch.Tensor:
    """
    Convert corners [B, M, 4, 2] => bounding boxes [B, M, 4].
    We compute min/max of x and y to get (xmin, ymin, xmax, ymax).
    """
    x_coords = corners[..., 0]
    y_coords = corners[..., 1]

    xmin = x_coords.min(dim=-1)[0]
    xmax = x_coords.max(dim=-1)[0]
    ymin = y_coords.min(dim=-1)[0]
    ymax = y_coords.max(dim=-1)[0]

    boxes = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
    return boxes


class BatchKorniaTransformWrapper:
    """
    Applies Kornia transforms on a batch of images + bounding boxes + keypoints, all on GPU.

    Expected Input:
      - images: a list of PyTorch tensors [C,H,W], or a single stacked tensor [B,C,H,W].
        Typically you'd pass a single stacked tensor on GPU, shape [B,C,H,W].
      - targets: a list of dictionaries, each with:
          'boxes': shape [M,4] or [1,4]
          'labels': ...
          'keypoints': shape [1, N, 3] or [N,3]

    Steps:
      1) Stack images into [B,C,H,W] if they're not already.
      2) Convert each target's 'boxes' to corners => [B, M, 4,2],
         split keypoints => coords [B,N,2] and visibility [B,N].
      3) Call `kornia_transform(...)` on (images, keypoints_xy, corners).
      4) Convert corners -> boxes, re-attach visibility, do out-of-bounds checks.
      5) Return (aug_imgs, aug_targets) with updated boxes, keypoints, etc.
    """

    def __init__(self, kornia_transform: AugmentationSequential):
        self.transform = kornia_transform

    def __call__(self,
                 images: torch.Tensor,
                 targets: List[Dict[str, Any]]
                 ) -> (torch.Tensor, List[Dict[str, Any]]):
        """
        Args:
            images: A tensor [B,C,H,W] on GPU (or CPU)
            targets: a list of B dictionaries (one per image).

        Returns:
            (aug_imgs, aug_targets):
              aug_imgs: [B,C,H',W'] after Kornia transforms
              aug_targets: a list of B dicts with updated 'boxes' and 'keypoints'
        """
        device = images.device
        B = images.shape[0]

        # 1) Convert each target's boxes + keypoints to a batch representation
        #    We'll assume each target has 'boxes' shape [M,4], 'keypoints' shape [1,N,3], etc.
        #    For simplicity, let's assume M=1 bounding box (like a single chessboard), or adapt if needed.

        all_boxes_list = []
        all_kp_xy_list = []
        all_vis_list = []
        max_keypoints = 0  # track the max number of keypoints among the batch, for stacking

        for t in targets:
            boxes_t = t["boxes"]  # shape [M,4] or [1,4]
            keypoints_3d = t["keypoints"]  # shape [1,N,3] or [N,3]
            if keypoints_3d.dim() == 3:
                keypoints_3d = keypoints_3d.squeeze(0)  # => [N,3] if it was [1,N,3]

            # split keypoints => coords [N,2], visibility [N]
            coords_xy = keypoints_3d[..., :2]
            visibility = keypoints_3d[..., 2]

            all_boxes_list.append(boxes_t)
            all_kp_xy_list.append(coords_xy)
            all_vis_list.append(visibility)

            max_keypoints = max(max_keypoints, coords_xy.shape[0])

        # We'll unify them into a single shape if you want to do a single transform call
        # But each image can have different # of keypoints => we might have to pad
        # For simplicity, assume same # of keypoints (like 4 corners).

        # 2) Stack bounding boxes => shape [B, M, 4]. Convert to corners => [B, M, 4,2].
        # If each image has 1 box (the chessboard), we get shape [B,1,4].
        batch_boxes = torch.stack(all_boxes_list, dim=0)  # => [B, M, 4]
        batch_corners = boxes_to_corners(batch_boxes)  # => [B, M, 4,2]

        # 3) Stack keypoints => shape [B, N, 2], visibility => shape [B, N].
        # We'll do a simple approach: assume each has the same # of keypoints
        # If not, you'd have to pad them to the same length.
        # For demonstration, let's just stack directly:
        batch_kp_xy = torch.stack(all_kp_xy_list, dim=0)  # [B, N, 2]
        batch_kp_vis = torch.stack(all_vis_list, dim=0)  # [B, N]

        # Move them to the same device
        batch_corners = batch_corners.to(device)
        batch_kp_xy = batch_kp_xy.to(device)
        batch_kp_vis = batch_kp_vis.to(device)

        # 4) Apply the Kornia transform => (aug_imgs, aug_kp, aug_corners)
        aug_imgs, aug_kp_xy, aug_corners = self.transform(images, batch_kp_xy, batch_corners)
        # shapes:
        #   aug_imgs => [B, C, H', W']
        #   aug_kp_xy => [B, N, 2]
        #   aug_corners => [B, M, 4,2]

        # 5) Convert corners -> boxes => shape [B,M,4].
        aug_boxes = corners_to_boxes(aug_corners)

        # 6) Re-attach visibility => shape [B, N, 3], do out-of-bounds checks
        B2, _, newH, newW = aug_imgs.shape
        assert B == B2, "Batch size mismatch after transform."

        final_targets = []
        for i in range(B):
            kp_xy_i = aug_kp_xy[i]  # => shape [N,2]
            vis_i = batch_kp_vis[i].clone()  # => shape [N]

            # out-of-bounds => set visibility=0
            for n in range(kp_xy_i.shape[0]):
                if vis_i[n] > 0:
                    x_coord, y_coord = kp_xy_i[n]
                    if (x_coord < 0 or x_coord >= newW or
                            y_coord < 0 or y_coord >= newH):
                        vis_i[n] = 0

            # Rebuild => [N,3]
            kps_3 = torch.cat([kp_xy_i, vis_i.unsqueeze(-1)], dim=-1)  # shape [N,3]

            # shape [M,4] for boxes
            boxes_i = aug_boxes[i]

            # Rebuild a new target dict
            old_t = targets[i]  # original references
            new_t = {}
            new_t['labels'] = old_t['labels'].to(device)
            new_t['boxes'] = boxes_i  # [M,4]
            new_t['keypoints'] = kps_3.unsqueeze(0)  # if you want [1,N,3]

            final_targets.append(new_t)

        return aug_imgs, final_targets

# Code for single image augmentation
class KorniaTransformWrapper:
    """
    A wrapper to apply Kornia augmentations to a single (image, target) pair:
      - image: [C, H, W]
      - target: dict with:
          'keypoints': [1, N, 3] => (x, y, visibility)
          'boxes': [1, 4]        => (xmin, ymin, xmax, ymax)
      - We expand dims to [1, C, H, W] so Kornia can treat it as a batch of size 1.
      - Split keypoints into coords [1, N, 2] and visibility [1, N].
      - Convert boxes to corners => [1, M, 4, 2].
      - Apply the augmentations => (aug_img, aug_coords, aug_corners).
      - Convert corners back to boxes => [1, M, 4].
      - Re-attach visibility => [1, N, 3], optionally set vis=0 if out-of-bounds.
    """

    def __init__(self, kornia_transform: AugmentationSequential):
        self.kornia_transform = kornia_transform

    def __call__(self, img_tensor: torch.Tensor, target: Dict[str, Any]):
        """
        Args:
            img_tensor: [C, H, W]  a single image
            target: {
                'keypoints': [1, N, 3],
                'boxes': [1, 4],
                ...
            }
        Returns:
            (aug_img, aug_target) with updated 'keypoints' and 'boxes'.
        """
        device = img_tensor.device  # Usually CPU if you haven't moved it, but can be GPU

        # 1) Expand image to batch => [1, C, H, W]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2) Keypoints => shape [1, N, 3]
        keypoints_3d = target["keypoints"]  # e.g. [1, N, 3]
        if keypoints_3d.dim() == 2:
            keypoints_3d = keypoints_3d.unsqueeze(0)

        # Split (x, y) from visibility
        coords_xy = keypoints_3d[..., :2]  # => [1, N, 2]
        visibility = keypoints_3d[..., 2]  # => [1, N]

        # 3) Boxes => [1, M, 4], convert to corners => [1, M, 4, 2]
        boxes_2d = target["boxes"]  # [1, 4]
        if boxes_2d.dim() == 2:
            boxes_2d = boxes_2d.unsqueeze(0)  # => [1, 1, 4] if single box
        corners = boxes_to_corners(boxes_2d)

        # Move to device
        img_tensor = img_tensor.to(device)
        coords_xy = coords_xy.to(device)
        visibility = visibility.to(device)
        corners = corners.to(device)

        # 4) Apply Kornia pipeline
        aug_img, aug_xy, aug_corners = self.kornia_transform(img_tensor, coords_xy, corners)

        # Now shapes:
        #   aug_img => [1, C, H', W']
        #   aug_xy => [1, N, 2]
        #   aug_corners => [1, M, 4, 2]

        # 5) Convert corners => boxes => [1, M, 4]
        aug_boxes = corners_to_boxes(aug_corners)

        # 6) Remove batch dimension from image => [C, H', W']
        aug_img = aug_img.squeeze(0)

        # 7) Re-attach visibility => shape [1, N, 3], possibly set v=0 if out-of-bounds
        B, _, H_out, W_out = aug_img.unsqueeze(0).shape  # effectively (1, C, H', W') => (1,H',W')

        final_keypoints_list = []
        # We'll assume we only have 1 in the batch dimension, but let's do the loop anyway
        for b_idx in range(B):
            kp_xy_b = aug_xy[b_idx]       # => [N, 2]
            vis_b   = visibility[b_idx]  # => [N]
            # out-of-bounds check
            for i in range(kp_xy_b.shape[0]):
                if vis_b[i] > 0:
                    x_coord, y_coord = kp_xy_b[i]
                    if x_coord < 0 or x_coord >= W_out or y_coord < 0 or y_coord >= H_out:
                        vis_b[i] = 0
            # Rebuild => [N, 3]
            kps_3 = torch.cat([kp_xy_b, vis_b.unsqueeze(-1)], dim=-1)
            final_keypoints_list.append(kps_3)

        final_keypoints = torch.stack(final_keypoints_list, dim=0)  # => [1, N, 3]

        # 8) Update target
        # - boxes shape => [1, M, 4], remove extra batch if needed
        aug_boxes = aug_boxes.squeeze(0)  # => [M,4]
        final_target = {
            "boxes": aug_boxes.unsqueeze(0),      # back to [1, M, 4]
            "keypoints": final_keypoints
        }

        # If you have other fields (labels, etc.), copy them
        if "labels" in target:
            final_target["labels"] = target["labels"].to(device)

        return aug_img, final_target