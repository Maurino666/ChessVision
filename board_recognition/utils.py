import torch

def reorder_corners_row_column(kp_xy: torch.Tensor,
                               vis: torch.Tensor,
                               top_left_origin: bool = True):
    """
    Robustly reorder four corners -> [TL, TR, BL, BR].

    kp_xy : [4, 2]   (x, y) pixel coords
    vis   : [4]      visibility flags

    Works for any perspective as long as the board projects to a convex quad.
    """
    if kp_xy.shape[0] != 4:
        return kp_xy, vis        # nothing to do

    # 1) sort by y   (remember: y grows downwards)
    sorted_idx = torch.argsort(kp_xy[:, 1])      # 0,1 = top row ; 2,3 = bottom row
    top_idx, bottom_idx = sorted_idx[:2], sorted_idx[2:]

    # 2) sort each row by x
    top_idx    = top_idx[torch.argsort(kp_xy[top_idx, 0])]
    bottom_idx = bottom_idx[torch.argsort(kp_xy[bottom_idx, 0])]

    ordered_idx = torch.cat([top_idx, bottom_idx], dim=0)   # TL, TR, BL, BR
    return kp_xy[ordered_idx], vis[ordered_idx]

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
