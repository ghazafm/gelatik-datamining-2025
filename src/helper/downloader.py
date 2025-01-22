from __future__ import annotations
from pathlib import Path
import requests
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

    # Download and save images
    if data_url:
        print(f"Downloading images from {data_url}...")
        zip_path = root / "images.zip"
        if "drive.google.com" in data_url:
            download_from_gdrive(data_url, zip_path)
        else:
            # Handle other download URLs here
            pass
        print("Extracting images...")
        extract_zip(output_dir, zip_path)

    # Download and save annotations
    if annotation_url:
        print(f"Downloading annotations from {annotation_url}...")
        annotation_path = annotation_file
        if "drive.google.com" in annotation_url:
            download_from_gdrive(annotation_url, annotation_path)
        else:
            # Handle other annotation URL downloading
            pass


def download_from_gdrive(gdrive_url, destination):
    """Download from Google Drive using file ID."""
    file_id = gdrive_url.split("/d/")[1].split("/")[0]
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(download_url, stream=True, timeout=10)
    response.raise_for_status()
    with open(destination, "wb") as f:
        f.write(response.content)
