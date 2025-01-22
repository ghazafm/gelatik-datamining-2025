from collections.abc import Callable
from PIL import Image


def transform_image(img: Image, transform: Callable) -> Image:
    """Apply the image transformation."""
    return transform(img)
