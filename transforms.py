from typing import Dict, Any, List
import torch
import kornia.augmentation as K
from kornia.augmentation import AugmentationSequential
from utils import (
    boxes_to_corners,
    corners_to_boxes,
    reorder_corners_by_quadrant
)

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


class BatchKorniaTransformWrapper:
    """
    Applies Kornia transforms on a batch of images + bounding boxes + keypoints, all on GPU.

    Expected Input:
      - images: a tensor [B,C,H,W] on GPU (or CPU)
      - targets: a list of B dictionaries (one per image), each with:
          'boxes': [M,4]  (chessboard bounding boxes)
          'labels': ...
          'keypoints': [1,N,3] or [N,3] (the corners, in x,y,visibility)

    Steps:
      1) Convert each target's 'boxes' => corners => [B,M,4,2],
         plus split keypoints => coords [B,N,2] and visibility [B,N].
      2) Call the Kornia transform on (images, coords, corners).
      3) Convert corners => boxes, re-attach visibility with out-of-bounds checks.
      4) Reorder corners via quadrant logic => [TL, TR, BL, BR].
      5) Return (aug_imgs, aug_targets).
    """

    def __init__(self, kornia_transform: AugmentationSequential):
        self.transform = kornia_transform

    def __call__(self,
                 images: torch.Tensor,
                 targets: List[Dict[str, Any]]
                 ) -> (torch.Tensor, List[Dict[str, Any]]):
        device = images.device
        B = images.shape[0]

        # Collect and stack boxes/keypoints from each target
        all_boxes_list = []
        all_kp_xy_list = []
        all_vis_list = []

        for t in targets:
            # boxes: [M,4] or [1,4]
            boxes_t = t["boxes"]
            # keypoints: [1,N,3] or [N,3]
            kps_3d = t["keypoints"]
            if kps_3d.dim() == 3:
                # squeeze if shape is [1,N,3]
                kps_3d = kps_3d.squeeze(0)  # => [N,3]

            coords_xy = kps_3d[..., :2]  # => [N,2]
            visibility = kps_3d[..., 2]  # => [N]

            all_boxes_list.append(boxes_t)
            all_kp_xy_list.append(coords_xy)
            all_vis_list.append(visibility)

        # Stack into [B, M, 4]
        batch_boxes = torch.stack(all_boxes_list, dim=0)   # => [B,M,4]
        # Convert boxes => corners => [B,M,4,2]
        batch_corners = boxes_to_corners(batch_boxes)

        # Stack keypoints => [B,N,2], [B,N]
        batch_kp_xy = torch.stack(all_kp_xy_list, dim=0)   # => [B,N,2]
        batch_kp_vis = torch.stack(all_vis_list, dim=0)    # => [B,N]

        # Move everything to device
        batch_corners = batch_corners.to(device)
        batch_kp_xy = batch_kp_xy.to(device)
        batch_kp_vis = batch_kp_vis.to(device)

        # Apply Kornia transforms => (aug_imgs, aug_kp_xy, aug_corners)
        aug_imgs, aug_xy, aug_corners = self.transform(images, batch_kp_xy, batch_corners)

        # Convert corners back => boxes => [B,M,4]
        aug_boxes = corners_to_boxes(aug_corners)

        # Build final targets with updated coords
        B2, _, newH, newW = aug_imgs.shape
        assert B == B2, "Mismatch in batch size after transforms"

        final_targets = []
        for i in range(B):
            kp_xy_i = aug_xy[i]           # => [N,2]
            vis_i = batch_kp_vis[i].clone()  # => [N]

            # Out-of-bounds => vis=0
            for n in range(kp_xy_i.shape[0]):
                if vis_i[n] > 0:
                    x_coord, y_coord = kp_xy_i[n]
                    if (x_coord < 0 or x_coord >= newW or
                        y_coord < 0 or y_coord >= newH):
                        vis_i[n] = 0

            # Reorder corners by quadrant => [TL, TR, BL, BR] if N=4
            reordered_xy, reordered_vis = reorder_corners_by_quadrant(
                kp_xy_i, vis_i, top_left_origin=True
            )

            # Rebuild => [N,3]
            kps_3 = torch.cat([reordered_xy, reordered_vis.unsqueeze(-1)], dim=-1)

            # Boxes => shape [M,4]
            boxes_i = aug_boxes[i]

            # Build a new target dict
            old_t = targets[i]
            new_t = {}
            new_t['labels'] = old_t['labels'].to(device)
            new_t['boxes'] = boxes_i
            # keep keypoints in shape [1,N,3] if that’s your standard
            new_t['keypoints'] = kps_3.unsqueeze(0)

            final_targets.append(new_t)

        return aug_imgs, final_targets


class KorniaTransformWrapper:
    """
    Applies Kornia transforms to a single image + target.

    Steps:
      1) Expand image => [1,C,H,W].
      2) Convert target boxes => corners => [1,M,4,2],
         split keypoints => [1,N,2], visibility => [1,N].
      3) Apply Kornia transform => get (aug_img, aug_xy, aug_corners).
      4) Convert corners => boxes; do out-of-bounds check => vis=0.
      5) Reorder corners => [TL, TR, BL, BR].
      6) Return updated (aug_img, target).
    """

    def __init__(self, kornia_transform: AugmentationSequential):
        self.kornia_transform = kornia_transform

    def __call__(self,
                 img_tensor: torch.Tensor,
                 target: Dict[str, Any]
                 ) -> (torch.Tensor, Dict[str, Any]):
        device = img_tensor.device

        # 1) Expand => [1,C,H,W]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)

        # 2) Extract keypoints => [1,N,3]
        kps_3d = target["keypoints"]
        if kps_3d.dim() == 2:
            kps_3d = kps_3d.unsqueeze(0)  # => [1,N,3]

        coords_xy = kps_3d[..., :2]  # => [1,N,2]
        visibility = kps_3d[..., 2]  # => [1,N]

        # Boxes => [1,M,4], corners => [1,M,4,2]
        boxes_2d = target["boxes"]
        if boxes_2d.dim() == 2:
            boxes_2d = boxes_2d.unsqueeze(0)  # => [1,M,4]
        corners = boxes_to_corners(boxes_2d)

        # Move to device
        img_tensor = img_tensor.to(device)
        coords_xy = coords_xy.to(device)
        visibility = visibility.to(device)
        corners = corners.to(device)

        # 3) Apply transforms => [1,C,H',W'], [1,N,2], [1,M,4,2]
        aug_img, aug_xy, aug_corners = self.kornia_transform(
            img_tensor, coords_xy, corners
        )

        # 4) Convert corners => boxes => [1,M,4]
        aug_boxes = corners_to_boxes(aug_corners)

        # Remove batch dimension from image => [C,H',W']
        aug_img = aug_img.squeeze(0)

        # Out-of-bounds => vis=0
        B, _, H_out, W_out = aug_img.unsqueeze(0).shape  # effectively (1,C,H',W')
        final_kp_list = []

        for b_idx in range(B):
            kp_xy_b = aug_xy[b_idx]  # => [N,2]
            vis_b = visibility[b_idx].clone()  # => [N]

            for i in range(kp_xy_b.shape[0]):
                if vis_b[i] > 0:
                    x_coord, y_coord = kp_xy_b[i]
                    if (x_coord < 0 or x_coord >= W_out or
                        y_coord < 0 or y_coord >= H_out):
                        vis_b[i] = 0

            # 5) Reorder corners => [TL, TR, BL, BR] if N=4
            reordered_xy, reordered_vis = reorder_corners_by_quadrant(
                kp_xy_b, vis_b, top_left_origin=True
            )

            # Rebuild => [N,3]
            final_kps_3 = torch.cat([reordered_xy, reordered_vis.unsqueeze(-1)], dim=-1)
            final_kp_list.append(final_kps_3)

        # final_keypoints => [1,N,3]
        final_keypoints = torch.stack(final_kp_list, dim=0)

        # 6) Build new target (boxes => [1,M,4], keypoints => [1,N,3])
        aug_boxes = aug_boxes.squeeze(0)  # => [M,4]
        new_target = {
            "boxes": aug_boxes.unsqueeze(0),  # => [1,M,4]
            "keypoints": final_keypoints
        }
        if "labels" in target:
            new_target["labels"] = target["labels"].to(device)

        return aug_img, new_target
