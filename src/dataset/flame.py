from __future__ import annotations
from pathlib import Path
from collections.abc import Callable
from dataset.base import BaseDataset
from dataset.base import BaseMultiDataset


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
            "data_url": "https://drive.google.com/uc?id=1BlteS5qxbBHIVjppHm8C7yp2PomQjdfl",
            "annotation_url": "https://drive.google.com/uc?id=10GY3l2GZI0UXQJIiyWbif7lx34n5Ylyw",
        }

    # pylint: enable=duplicate-code


class FlameFOV(BaseDataset):
    def __init__(
        self,
        root: str | Path = "../data/flame_fov",
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
            "data_url": "https://drive.google.com/uc?id=1hetnLFcleuBqqIlaeGNEPvU02K6Cv5gT",
            "annotation_url": "https://drive.google.com/uc?id=101SAnRrkZdNtYGE5rnbZEk0SiPEsjAyo",
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
            "data_url": "https://drive.google.com/uc?id=1vx6zqxoaCS5eqVu3AqbjiWhp4uTa5_16",
            "annotation_url": "https://drive.google.com/uc?id=10LIC4QZhqx9GTKgrhkMdNdpJ8VBYuMk4",
        }

    # pylint: enable=duplicate-code


class FlameSatelite(BaseMultiDataset):
    def __init__(
        self,
        root: str | Path = "../data/flame_satellite",
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
            "data_url": "https://drive.google.com/uc?id=1QXWOIxrv4Q2CccWsafpDh7miASZqWDbY",
            "annotation_url": "https://drive.google.com/uc?id=1KXJzjXNguRnT59-XlQhLomT-xfsjTYis",
        }

    # pylint: enable=duplicate-code
