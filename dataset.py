import os
import csv
import torch
from PIL import Image
import torchvision.transforms.functional as F  # Assicurati di importare F.to_tensor

class ChessboardCornersDataset(torch.utils.data.Dataset):
    def __init__(self, root, csv_file, transforms=None):
        """
        root: path to the directory containing images
        csv_file: path to the CSV file with columns:
                  filename, xTL, yTL, xTR, yTR, xBL, yBL, xBR, yBR
        transforms: a callable/transform object applied on (image, target)
        """
        self.root = root
        self.transforms = transforms
        self.records = []

        # Read the CSV data into a list of dictionaries
        with open(csv_file, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.records.append(row)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        # Retrieve the row (dictionary) for this sample
        data = self.records[idx]

        # Image filename
        filename = data['filename']
        img_path = os.path.join(self.root, filename)

        # Load the image in RGB mode
        img = Image.open(img_path).convert("RGB")

        # Convert PIL Image to torch.Tensor [C, H, W]
        img = F.to_tensor(img)

        # Extract corner coordinates from CSV
        xTL, yTL = float(data['xTL']), float(data['yTL'])
        xTR, yTR = float(data['xTR']), float(data['yTR'])
        xBL, yBL = float(data['xBL']), float(data['yBL'])
        xBR, yBR = float(data['xBR']), float(data['yBR'])

        # Create the keypoints array
        keypoints = [[
            [xTL, yTL, 2],
            [xTR, yTR, 2],
            [xBL, yBL, 2],
            [xBR, yBR, 2]
        ]]
        keypoints = torch.as_tensor(keypoints, dtype=torch.float32)

        # Compute a bounding box that encloses the four corners
        xs = [xTL, xTR, xBL, xBR]
        ys = [yTL, yTR, yBL, yBR]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        boxes = torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32)

        labels = torch.tensor([1], dtype=torch.int64)  # single class (e.g. "chessboard")

        # Create the target dictionary
        target = {
            'boxes': boxes,
            'labels': labels,
            'keypoints': keypoints[:, :, :2],        # shape (1,4,2)
            'keypoints_visible': keypoints[:, :, 2],   # shape (1,4)
        }

        # Apply transformations if provided
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target
