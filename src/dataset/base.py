from __future__ import annotations
from pathlib import Path
from typing import Any
from collections.abc import Callable
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, UnidentifiedImageError
from helper.downloader import download_file
from helper.utils import check_integrity
from helper.image_processing import transform_image


class BaseDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        download: bool = False,
        urls: dict[str, str] | None = None,
    ):
        # Initialize dataset-specific URLs
        if urls is None:
            urls = self.get_default_urls()

        self.root = Path(root) if isinstance(root, str) else root
        self.dir = self.root / "images"
        self.annotation_file = self.root / "annotations.csv"
        self.transform = transform
        self.urls = urls

        # Download dataset if specified
        if download:
            download_file(
                self.root,
                self.dir,
                self.annotation_file,
                self.urls["data_url"],
                self.urls["annotation_url"],
            )

        # Check dataset integrity
        if not check_integrity(self.dir, self.annotation_file):
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download it."
            )

        # Prepare image paths and bounding boxes
        self.image_paths = []
        self.bounding_boxes = []

        # Load annotations
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            reader = np.loadtxt(
                file,
                delimiter=",",
                dtype=str,
                skiprows=1,
                usecols=[0, 2, 3, 4, 5],
            )
            for row in reader:
                img_path = self.dir / row[0]
                bbox = [(float(x) if x != "" else 0.0) for x in row[1:]]
                self.image_paths.append(img_path)
                self.bounding_boxes.append(bbox)

    def get_default_urls(self) -> dict[str, str]:
        """Override this method in the subclasses to provide dataset-specific URLs."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, torch.tensor]:
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


class BaseMultiDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
        download: bool = False,
        urls: dict[str, str] | None = None,
    ):
        # Initialize dataset-specific URLs
        if urls is None:
            urls = self.get_default_urls()

        self.root = Path(root) if isinstance(root, str) else root
        self.dir = self.root / "images"
        self.annotation_file = self.root / "annotations.csv"
        self.transform = transform
        self.urls = urls

        # Download dataset if specified
        if download:
            download_file(
                self.root,
                self.dir,
                self.annotation_file,
                self.urls["data_url"],
                self.urls["annotation_url"],
            )

        # Check dataset integrity
        if not check_integrity(self.dir, self.annotation_file):
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download it."
            )

        # Prepare image paths and bounding boxes
        self.image_paths = []
        self.bounding_boxes = []

        # Load annotations
        with open(self.annotation_file, "r", encoding="utf-8") as file:
            reader = np.loadtxt(
                file,
                delimiter=",",
                dtype=str,
                skiprows=1,
                usecols=[0, 2, 3, 4, 5],
            )
            for row in reader:
                img_path = self.dir / row[0]
                bbox = [(float(x) if x != "" else 0.0) for x in row[1:]]
                if img_path not in self.image_paths:
                    self.image_paths.append(img_path)
                    self.bounding_boxes.append([bbox])
                else:
                    idx = self.image_paths.index(img_path)
                    self.bounding_boxes[idx].append(bbox)


    def get_default_urls(self) -> dict[str, str]:
        """Override this method in the subclasses to provide dataset-specific URLs."""
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> tuple[Any, torch.tensor]:
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
