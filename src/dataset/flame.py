from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
from dataset.base import BaseDataset


class FlameRGB(BaseDataset):
    def __init__(
        self,
        root: str | Path = "../data/flame_rgb",
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
            "data_url": "https://drive.google.com/file/d/1BlteS5qxbBHIVjppHm8C7yp2PomQjdfl/view?usp=sharing",
            "annotation_url": "",
        }
    # pylint: enable=duplicate-code


class FlameFOV(BaseDataset):
    def __init__(
        self,
        root: str | Path = "../data/flame_thermal",
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
            "data_url": "https://drive.google.com/file/d/1VeHGZeunP1Hq_Qvf31MvB5FbL0CB5ZvV/view?usp=sharing",
            "annotation_url": "",
        }
    # pylint: enable=duplicate-code

class FlameThermal(BaseDataset):
    def __init__(
        self,
        root: str | Path = "../data/flame_thermal",
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
            "data_url": "https://drive.google.com/file/d/1vx6zqxoaCS5eqVu3AqbjiWhp4uTa5_16/view?usp=sharing",
            "annotation_url": "",
        }
    # pylint: enable=duplicate-code
