from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
from dataset.base import BaseDataset


class Albi(BaseDataset):
    def __init__(
        self,
        root: str | Path = "../data/albi",
        transform: Callable | None = None,
        download: bool = False,
        urls: dict[str, str] | None = None,
    ):
        super().__init__(
            root=root,
            transform=transform,
            download=download,
            urls=urls,
        )

    # pylint: disable=duplicate-code
    def get_default_urls(self) -> dict[str, str]:
        return {
            "data_url": "https://drive.google.com/file/d/1lEmbEJHXVn7z80fYevjMVdqBo4FdwLpP/view?usp=drive_link",
            "annotation_url": "https://drive.google.com/file/d/1cVCm9gTWR1ovbhb1lpO8Fx7NqqO7kIFb/view?usp=drive_link",
        }


# pylint: enable=duplicate-code
