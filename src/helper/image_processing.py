from PIL import Image
from typing import Callable


def transform_image(img: Image, transform: Callable) -> Image:
    """Apply the image transformation."""
    return transform(img)
