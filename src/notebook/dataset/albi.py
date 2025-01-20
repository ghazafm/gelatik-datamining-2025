import os
import csv
import torch
from torch.utils.data import Dataset
from typing import Any, Callable, Optional, Tuple, Union
import requests
from io import BytesIO, StringIO
from PIL import Image, UnidentifiedImageError
from pathlib import Path
import zipfile
import shutil


class Albi(Dataset):
    """Custom Dataset for the Albi dataset with images and bounding boxes.

    Args:
        root (str or Path): Root directory of dataset where the dataset will be stored.
        transform (callable, optional): A function/transform that takes in a PIL image and returns a transformed version.
        download (bool, optional): If True, downloads the dataset and annotations if not already present.
        img_url (str, optional): URL for downloading the image data (assumes it's a zip file).
        annotation_url (str, optional): URL for downloading the annotation CSV file.
    """

    def __init__(
        self,
        root: Union[str, Path] = "../../data/albi",
        transform: Optional[Callable] = None,
        download: bool = False,
        img_url: Optional[
            str
        ] = "https://drive.google.com/file/d/1lEmbEJHXVn7z80fYevjMVdqBo4FdwLpP/view?usp=drive_link",
        annotation_url: Optional[
            str
        ] = "https://drive.google.com/file/d/1cVCm9gTWR1ovbhb1lpO8Fx7NqqO7kIFb/view?usp=drive_link",
    ):
        self.root = Path(root) if isinstance(root, str) else root
        self.img_dir = self.root / "images"
        self.annotation_file = self.root / "annotations.csv"
        self.transform = transform

        if download:
            self.download(img_url, annotation_url)

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted. Use download=True to download it."
            )

        self.image_paths = []
        self.bounding_boxes = []

        with open(self.annotation_file, "r") as file:
            reader = csv.reader(file)
            for row in reader:
                img_path = self.img_dir / row[0]
                bbox = [float(x) for x in row[1:]]
                self.image_paths.append(img_path)
                self.bounding_boxes.append(bbox)

    def __len__(self) -> int:
        """Return the total number of images."""
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        """Retrieve an image and its corresponding bounding box."""
        img_path = self.image_paths[idx]
        bbox = torch.tensor(self.bounding_boxes[idx])

        try:
            img = Image.open(img_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

        except UnidentifiedImageError as e:
            print(f"Error loading image from {img_path}: {e}")
            return None, None

        return img, bbox

    def _check_integrity(self) -> bool:
        """Check if dataset exists and is complete."""
        return self.img_dir.exists() and self.annotation_file.exists()

    def download(self, img_url: Optional[str], annotation_url: Optional[str]) -> None:
        """Download and extract dataset if not already present."""
        if self._check_integrity():
            print("Dataset already exists. Skipping download.")
            return

        self.img_dir.mkdir(parents=True, exist_ok=True)

        def download_from_gdrive(gdrive_url, destination):
            file_id = gdrive_url.split("/d/")[1].split("/")[0]
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            response = requests.get(download_url, stream=True)
            response.raise_for_status()
            with open(destination, "wb") as f:
                f.write(response.content)

        # Download and save images
        if img_url:
            print(f"Downloading images from {img_url}...")
            zip_path = self.root / "images.zip"
            if "drive.google.com" in img_url:
                download_from_gdrive(img_url, zip_path)
            else:
                response = requests.get(img_url, stream=True)
                response.raise_for_status()
                with open(zip_path, "wb") as f:
                    f.write(response.content)
            print("Extracting images...")
            self.extract_zip(zip_path)

        # Download and save annotations
        if annotation_url:
            print(f"Downloading annotations from {annotation_url}...")
            annotation_path = self.annotation_file
            if "drive.google.com" in annotation_url:
                download_from_gdrive(annotation_url, annotation_path)
            else:
                response = requests.get(annotation_url)
                response.raise_for_status()
                with open(annotation_path, "w") as f:
                    f.write(response.text)

    def extract_zip(self, zip_path: Path) -> None:
        """Extract the zip file and check if it contains folders or more zip files."""
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.img_dir)

        self.cleanup_folder_structure()

        zip_path.unlink()

    def cleanup_folder_structure(self) -> None:
        """Check if there is a nested images folder and flatten the structure."""
        for item in os.listdir(self.img_dir):
            item_path = self.img_dir / item
            if item_path.is_dir():
                if item == "images":
                    print(f"Found a nested 'images' folder. Flattening structure...")
                    self.flatten_images_folder(item_path)
                else:
                    print(f"Found a directory: {item}. Skipping...")

            # Remove unwanted __MACOSX folder
            if item == "__MACOSX":
                print(f"Removing unwanted __MACOSX folder...")
                macosx_dir = self.img_dir / item
                shutil.rmtree(macosx_dir)

    def flatten_images_folder(self, nested_folder_path: Path) -> None:
        """Move contents from the nested images folder to the parent directory."""
        for item in os.listdir(nested_folder_path):
            item_path = nested_folder_path / item
            target_path = self.img_dir / item
            if item_path.is_dir():
                for sub_item in os.listdir(item_path):
                    sub_item_path = item_path / sub_item
                    sub_item_path.rename(self.img_dir / sub_item)
                os.rmdir(item_path)
            else:
                item_path.rename(target_path)

    def extra_repr(self) -> str:
        """Additional info for printing."""
        return f"Root: {self.root}, Transform: {self.transform}"
