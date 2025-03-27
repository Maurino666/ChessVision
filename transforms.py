import torchvision.transforms.functional as F

class CustomResize:
    def __init__(self, size):
        """
        size: (new_width, new_height) or an integer for square resizing
        """
        if isinstance(size, int):
            # If an integer is given, we assume a square
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, target):
        # img is a PIL Image
        # target is a dict containing 'keypoints' and 'boxes' among others.

        old_w, old_h = img.size  # Original width, height from PIL

        # We assume self.size is (width, height),
        # but be careful with the order, it could also be (height, width) depending on usage.
        new_w, new_h = self.size

        # 1) Resize the image
        img_resized = F.resize(img, (new_h, new_w))

        # 2) Compute the scaling factors
        scale_x = new_w / old_w
        scale_y = new_h / old_h

        # 3) Update keypoints:
        # shape (N, K, 2), each keypoint is (x, y)
        keypoints = target['keypoints'].clone()
        keypoints[..., 0] *= scale_x  # scale x-coordinates
        keypoints[..., 1] *= scale_y  # scale y-coordinates
        target['keypoints'] = keypoints

        # 4) Update bounding boxes [xmin, ymin, xmax, ymax]
        boxes = target['boxes'].clone()
        boxes[..., [0, 2]] *= scale_x
        boxes[..., [1, 3]] *= scale_y
        target['boxes'] = boxes

        return img_resized, target
