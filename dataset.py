import os
import csv
import torch
import torch.utils.data
from PIL import Image
import torchvision.transforms.functional as F


class ChessboardCornersDataset(torch.utils.data.Dataset):
    """
    A custom Dataset to detect chessboard corners:
      - Reads a CSV file containing corner coordinates (filename, xTL, yTL, etc.)
      - Loads images from disk
      - Resizes each image to (new_height, new_width)
      - Scales the keypoints and bounding boxes accordingly
      - Returns (image_tensor, target_dict)

    The output image shape will be (C, new_height, new_width).
    """

    def __init__(self, root, csv_file, resize_hw=(800, 800)):
        """
        Args:
            root (str): Path to the directory containing images.
            csv_file (str): Path to the CSV file with columns
                            (filename, xTL, yTL, vTL, xTR, yTR, vTR, xBL, yBL, vBL, xBR, yBR, vBR).
            resize_hw (tuple): The (height, width) to resize every image to.
                               For instance, (800, 800).
        """
        self.root = root
        self.csv_file = csv_file
        self.resize_hw = resize_hw  # (new_height, new_width)
        self.records = []

        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        """
        Reads one record from the CSV, loads and resizes the image,
        scales the bounding box and keypoints, and returns them.
        """
        data = self.records[idx]
        filename = data['filename']
        img_path = os.path.join(self.root, filename)

        # Load image in RGB mode
        img = Image.open(img_path).convert("RGB")

        # Original size
        orig_width, orig_height = img.size

        # Target size
        new_height, new_width = self.resize_hw

        # Resize the image (CPU side). This might distort aspect ratio.
        img = F.resize(img, (new_height, new_width))

        # Convert to a PyTorch tensor => shape [C, new_height, new_width]
        img = F.to_tensor(img)

        # Retrieve corners from CSV
        xTL, yTL, vTL = float(data['xTL']), float(data['yTL']), int(data['vTL'])
        xTR, yTR, vTR = float(data['xTR']), float(data['yTR']), int(data['vTR'])
        xBL, yBL, vBL = float(data['xBL']), float(data['yBL']), int(data['vBL'])
        xBR, yBR, vBR = float(data['xBR']), float(data['yBR']), int(data['vBR'])

        # Compute scale factors for width/height
        scale_x = new_width / orig_width
        scale_y = new_height / orig_height

        # Scale corners
        xTL_new, yTL_new = xTL * scale_x, yTL * scale_y
        xTR_new, yTR_new = xTR * scale_x, yTR * scale_y
        xBL_new, yBL_new = xBL * scale_x, yBL * scale_y
        xBR_new, yBR_new = xBR * scale_x, yBR * scale_y

        keypoints = [[
            [xTL_new, yTL_new, vTL],
            [xTR_new, yTR_new, vTR],
            [xBL_new, yBL_new, vBL],
            [xBR_new, yBR_new, vBR]
        ]]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        # Build a bounding box from the scaled corners
        xs = [xTL_new, xTR_new, xBL_new, xBR_new]
        ys = [yTL_new, yTR_new, yBL_new, yBR_new]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        labels = torch.tensor([1], dtype=torch.int64)  # single class (e.g., 1 for "chessboard")

        target = {
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints
        }

        return img, target

