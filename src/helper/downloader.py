from __future__ import annotations
from pathlib import Path
import gdown
from helper.utils import check_integrity, extract_zip


def download_file(
    root: Path,
    output_dir: Path,
    annotation_file: Path,
    data_url: str | None = None,
    annotation_url: str | None = None,
) -> None:
    if check_integrity(output_dir, annotation_file):
        print("Dataset already exists. Skipping download.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    if data_url:
        print(f"Downloading images from {data_url}...")
        zip_path = root / "images.zip"
        if "drive.google.com" in data_url:
            download_from_gdrive(data_url, zip_path)
        else:
            pass
        print("Extracting images...")
        extract_zip(output_dir, zip_path)

    if annotation_url:
        print(f"Downloading annotations from {annotation_url}...")
        annotation_path = annotation_file
        if "drive.google.com" in annotation_url:
            download_from_gdrive(annotation_url, annotation_path)
        else:
            print("url not supported")


def download_from_gdrive(gdrive_url: str, destination: Path) -> None:
    """Download from Google Drive using gdown."""
    # Use the gdown library to download from Google Drive
    gdown.download(gdrive_url, str(destination), quiet=False)
    print(f"Downloaded file to {destination}")
