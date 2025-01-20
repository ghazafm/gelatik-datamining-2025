import os
import csv
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple
import torchvision.transforms as transforms
import requests
from io import BytesIO
from PIL import Image, UnidentifiedImageError


class Flame(Dataset):
    """Custom Dataset for the Flame dataset with images and bounding boxes.

    Args:
        img_dir (str): Directory with all the images.
        annotation_file (str): Path to the CSV file with annotations.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
    """

    def __init__(
        self, img: str, annotation_file: str, transform: Optional[Callable] = None
    ):
        self.img_dir = img
        self.annotation_file = annotation_file
        self.transform = transform

        self.image_paths = []
        self.bounding_boxes = []
        self.isurl = False

        with open(annotation_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                img_path = row[0]
                bbox = [float(x) for x in row[1:]]
                self.image_paths.append(img_path)
                self.bounding_boxes.append(bbox)

        if isinstance(img, list):
            if img[0].startswith("http") or self.img_dir.startswith("https"):
                self.isurl = True
        else:
            if img.startswith("http") or self.img_dir.startswith("https"):
                self.isurl = True

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Retrieve an image and its corresponding bounding box."""

        img_path = self.image_paths[idx]
        bbox = torch.tensor(self.bounding_boxes[idx])

        try:
            if self.isurl:
                img_url = self.img_dir[idx]
                response = requests.get(img_url)
                response.raise_for_status()  # Raise an error for bad HTTP responses
                img = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                img_path = os.path.join(self.img_dir, img_path)
                img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

        except (UnidentifiedImageError, requests.exceptions.RequestException) as e:
            print(f"Error loading image from {img_path}: {e}")
            # You can handle it here, e.g., skip this sample or use a default image
            return None, None  # Return None for error cases (or handle differently)

        return img, bbox
