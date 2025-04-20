import torch

def reorder_corners_by_quadrant(kp_xy: torch.Tensor,
                                vis: torch.Tensor,
                                top_left_origin: bool = True
                                ) -> (torch.Tensor, torch.Tensor):
    """
    Reorders four corners into [TL, TR, BL, BR] based on their position
    relative to the center. Assumes kp_xy.shape == [N, 2], vis.shape == [N],
    typically N=4 for a single chessboard.

    If top_left_origin=True => y increases downwards (standard image coordinates).
      => 'Top' means smaller y, 'Bottom' means bigger y.

    Returns:
      - new_kp_xy: [4, 2]
      - new_vis:   [4]
    """
    N = kp_xy.shape[0]
    # If we don't have exactly 4 corners, just skip reordering.
    if N != 4:
        return kp_xy, vis

    # 1) Compute center
    cx = kp_xy[:, 0].mean()
    cy = kp_xy[:, 1].mean()

    # 2) Classify each corner in TL/TR/BL/BR
    corner_dict = {}
    for i in range(N):
        x, y = kp_xy[i]
        dx = x - cx
        dy = y - cy

        # In top-left origin:
        #   top => y < cy
        #   bottom => y >= cy
        #   left => x < cx
        #   right => x >= cx
        if dy < 0 and dx < 0:
            corner_dict["TL"] = (kp_xy[i], vis[i])
        elif dy < 0 and dx >= 0:
            corner_dict["TR"] = (kp_xy[i], vis[i])
        elif dy >= 0 and dx < 0:
            corner_dict["BL"] = (kp_xy[i], vis[i])
        else:
            corner_dict["BR"] = (kp_xy[i], vis[i])

    # 3) Build final arrays [TL, TR, BL, BR].
    #    If a corner is missing from a quadrant, we default to (0,0) with vis=0.
    final_xy = []
    final_vis = []
    for key in ["TL", "TR", "BL", "BR"]:
        if key in corner_dict:
            c_xy, c_vis = corner_dict[key]
        else:
            c_xy = torch.tensor([0.0, 0.0], device=kp_xy.device)
            c_vis = torch.tensor(0, device=kp_xy.device)
        final_xy.append(c_xy)
        final_vis.append(c_vis)

    new_kp_xy = torch.stack(final_xy, dim=0)  # => [4,2]
    new_vis   = torch.stack(final_vis, dim=0) # => [4]
    return new_kp_xy, new_vis

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
