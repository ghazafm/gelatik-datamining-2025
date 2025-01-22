import csv
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple, Union
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from helper.downloader import download_file
from helper.utils import check_integrity
from helper.image_processing import transform_image


class Flame3(Dataset):
    def __init__(
        self,
        root: Union[str, Path] = "../data/flame3",
        transform: Optional[Callable] = None,
        download: bool = False,
        data_url: Optional[
            str
        ] = "https://drive.google.com/file/d/1lEmbEJHXVn7z80fYevjMVdqBo4FdwLpP/view?usp=drive_link",
        annotation_url: Optional[
            str
        ] = "https://drive.google.com/file/d/1cVCm9gTWR1ovbhb1lpO8Fx7NqqO7kIFb/view?usp=drive_link",
    ):
        self.root = Path(root) if isinstance(root, str) else root
        self.dir = self.root / "images"
        self.annotation_file = self.root / "annotations.csv"
        self.transform = transform

        if download:
            download_file(
                self.root, self.dir, self.annotation_file, data_url, annotation_url
            )

        if not check_integrity(self.dir, self.annotation_file):
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download it."
            )

        self.image_paths = []
        self.bounding_boxes = []

        with open(self.annotation_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                img_path = self.dir / row[0]
                bbox = [float(x) for x in row[1:]]
                self.image_paths.append(img_path)
                self.bounding_boxes.append(bbox)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        img_path = self.image_paths[idx]
        bbox = torch.tensor(self.bounding_boxes[idx])

        try:
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = transform_image(img, self.transform)

        except UnidentifiedImageError as e:
            print(f"Error loading image from {img_path}: {e}")
            return None, None
        return img, bbox
