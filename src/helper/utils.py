import os
import shutil
from pathlib import Path


def check_integrity(img_dir, annotation_file):
    """Check if dataset exists and is complete."""
    return img_dir.exists() and annotation_file.exists()


def cleanup_folder_structure(img_dir):
    """Flatten any nested directories or unwanted files."""
    for item in os.listdir(img_dir):
        item_path = img_dir / item
        if item_path.is_dir():
            if item == "images":
                print(f"Found a nested 'images' folder. Flattening structure...")
                flatten_images_folder(item_path)
            else:
                print(f"Found a directory: {item}. Skipping...")

        # Remove unwanted __MACOSX folder
        if item == "__MACOSX":
            print(f"Removing unwanted __MACOSX folder...")
            macosx_dir = img_dir / item
            shutil.rmtree(macosx_dir)


def flatten_images_folder(nested_folder_path):
    """Move contents from the nested images folder to the parent directory."""
    for item in os.listdir(nested_folder_path):
        item_path = nested_folder_path / item
        target_path = nested_folder_path.parent / item
        item_path.rename(target_path)
    os.rmdir(nested_folder_path)


def extract_zip(dir, zip_path: Path) -> None:
    import zipfile

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(dir)

    cleanup_folder_structure(dir)
    zip_path.unlink()


def extra_repr(root, transform) -> str:
    return f"Root: {root}, Transform: {transform}"
