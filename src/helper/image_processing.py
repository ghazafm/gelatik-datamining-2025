from collections.abc import Callable
from PIL import Image
from torchvision.transforms import functional as F


def transform_image(img: Image, transform: Callable) -> Image:
    """Apply the image transformation."""
    return transform(img)


def calculate_padding(image):
    w, h = image.size
    max_wh = max(w, h)
    pad_w = (max_wh - w) // 2
    pad_h = (max_wh - h) // 2
    return (pad_w, pad_h, max_wh - w - pad_w, max_wh - h - pad_h)


class SquarePadTransform:
    def __call__(self, image):
        padding = calculate_padding(image)
        return F.pad(image, padding, fill=0, padding_mode="constant")
