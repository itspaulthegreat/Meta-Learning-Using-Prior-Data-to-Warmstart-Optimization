"""
Data needed for the project. The data is not included in the repository due to its size.
The original data can be downloaded from the following link: 
https://www.kaggle.com/datasets/coreylammie/deepweedsx

We split it for easier use of dataloaders.
We additionally provide a simplified version that does not have an imbalanced class for 
ease of testing your code.
Please download the processed data from:
ml.informatik.uni-freiburg.de/~biedenka/dwx_compressed.tar.gz
"""
from __future__ import annotations

from typing import Callable, Iterable
from typing_extensions import TypeAlias

import argparse
import tarfile
import logging
import os
import shutil
import urllib.request
from pathlib import Path

from torchvision import transforms
from torchvision.datasets import ImageFolder, FashionMNIST


logger = logging.getLogger(__name__)

HERE = Path(__file__).absolute().parent / "data"

Dimensions: TypeAlias = tuple[int, int, int]


DEEP_WOODS_LINK = "http://ml.informatik.uni-freiburg.de/~biedenka/dwx_compressed.tar.gz"

DEEP_WOODS_DEFAULT_DIR = HERE / "deepweedsx"
DEEP_WOODS_DEFAULT_TARBALL_PATH = DEEP_WOODS_DEFAULT_DIR / "processed_data.tar"


def _unpack_tarball(
    tarball: Path = DEEP_WOODS_DEFAULT_TARBALL_PATH,
    dest: Path = DEEP_WOODS_DEFAULT_DIR,
) -> None:
    dir_contents = list(dest.iterdir())
    if len(dir_contents) > 1:
        logging.debug(
            f"Already unpacked most likely with {len(dir_contents)} items"
            f" in {dest}."
            f"\n{dir_contents}"
        )
    logging.debug(f"Unpacking {tarball}")
    tar = tarfile.open(name=tarball, mode="r:gz")
    tar.extractall(path=dest)


def _download_deepweeds(
    url: str = DEEP_WOODS_LINK,
    dest: Path = DEEP_WOODS_DEFAULT_TARBALL_PATH
) -> Path:
    if dest.exists():
        logging.debug(f"Already found file at {dest}")
        return dest

    logging.debug(f"Downloading from {url} to {dest}")
    dest.parent.mkdir(exist_ok=True, parents=True)

    with urllib.request.urlopen(url) as response, open(dest, "wb") as f:
        shutil.copyfileobj(response, f)

    logging.debug(f"Download finished at {dest}")
    return dest


def load_deep_woods(
    datadir: Path,
    balanced: bool = False,
    resize: tuple[int, int] = (16, 16),
    transform: Iterable[Callable] = (),
    # TODO: Auto download and unpack
    download: bool = True,
) -> tuple[Dimensions, ImageFolder, ImageFolder]:
    """Load the DeepWeeds dataset.

    :param balanced:
        Whether to load the balanced dataset or not.
    :param resize:
        What to resize the image to
    :param transform:
        The transformation to apply to images.

    :return: The train and test datasets.
    """
    if download:
        tar_path = _download_deepweeds(dest=datadir / "deepweedsx_compressed.tar.gz")
        _unpack_tarball(tarball=tar_path, dest=datadir)

    # Original dimensions are 3x256x256
    img_width, img_height = resize
    dimensions = (3, img_width, img_height)

    pre_processing = transforms.Compose(
        [
            transforms.Resize((img_width, img_height)),
            transforms.ToTensor(),
            *transform,
        ]
    )

    suffix = "" if not balanced else "_balanced"

    train_path = datadir / f"train{suffix}"
    test_path = datadir / f"test{suffix}"

    train_dataset = ImageFolder(root=str(train_path), transform=pre_processing)
    test_dataset = ImageFolder(root=str(test_path), transform=pre_processing)

    return dimensions, train_dataset, test_dataset


def load_fashion_mnist(
    datadir: Path,
    transform: Iterable[Callable] = (),
) -> tuple[Dimensions, FashionMNIST, FashionMNIST]:
    img_width = 28
    img_height = 28
    dimensions = (1, img_width, img_height)

    pre_processing = transforms.Compose([transforms.ToTensor(), *transform])

    train = FashionMNIST(
        root=str(datadir),
        train=True,
        download=True,
        transform=pre_processing,
    )

    test = FashionMNIST(
        root=str(datadir),
        train=False,
        download=True,
        transform=pre_processing,
    )
    return dimensions, train, test


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MF example using BOHB.")
    parser.add_argument(
        "--base_dir",
        default=None,
        help="base directory where the data/ folder will be created for datasets",
    )
    args = parser.parse_args()

    # overwrite default data directory if a not None string input
    if args.base_dir is not None and isinstance(args.base_dir, str):
        HERE = Path(args.base_dir).absolute() / "data"

    logging.basicConfig(level=logging.DEBUG)

    load_deep_woods(datadir=HERE / "deepweedsx")
    load_fashion_mnist(datadir=HERE / "FashionMNIST")
