import os
import shutil
import zipfile
from pathlib import Path
import torch


def check_integrity(img_dir, annotation_file):
    """Check if dataset exists and is complete."""
    return img_dir.exists() and annotation_file.exists()


def cleanup_folder_structure(img_dir):
    """Flatten any nested directories or unwanted files."""
    for item in os.listdir(img_dir):
        item_path = img_dir / item
        if item_path.is_dir():
            if "__MACOSX" not in item:
                print("Found a nested 'images' folder. Flattening structure...")
                flatten_images_folder(item_path)
            else:
                print(f"Found a directory: {item}. Skipping...")

        # Remove unwanted __MACOSX folder
        if item == "__MACOSX":
            print("Removing unwanted __MACOSX folder...")
            macosx_dir = img_dir / item
            shutil.rmtree(macosx_dir)


def flatten_images_folder(nested_folder_path):
    """Move contents from the nested images folder to the parent directory."""
    for item in os.listdir(nested_folder_path):
        item_path = nested_folder_path / item
        target_path = nested_folder_path.parent / item
        item_path.rename(target_path)
    os.rmdir(nested_folder_path)


def extract_zip(output_dir, zip_path: Path) -> None:

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)

    cleanup_folder_structure(output_dir)
    zip_path.unlink()


def extra_repr(root, transform) -> str:
    return f"Root: {root}, Transform: {transform}"


def collate_fn(batch):
    images = []
    bboxes = []
    
    for img, bbox in batch:
        if img is not None:  # Skip corrupted images
            images.append(img)
            bboxes.append(bbox)
    
    # Stack images into a batch
    images = torch.stack(images, dim=0)
    
    # Return images and a list of bounding boxes (not stacked)
    return images, bboxes
