import errno
import numpy as np
import pathlib
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset

from .transforms import get_image_transform
from .utils import get_device


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, config, image_size, train):
        self.train = train
        transform = get_image_transform(image_size=image_size)

        super().__init__(
            str(config.data_path),
            train=self.train,
            download=True,
            transform=transform,
        )

    def __getitem__(self, item):
        return super().__getitem__(item)


def get_dataset(config: dict, name: str, image_size: int, testing: bool = False):
    if name == "MNIST":
        trainset, valset = MNISTDataset(config, image_size, train=True), MNISTDataset(
            config, image_size, train=False
        )

    classes = list(set(trainset.targets.numpy()))

    if testing:
        indices = np.arange(0, 20)
        trainset = Subset(trainset, indices)
        valset = Subset(valset, indices)

    return trainset, valset, classes


def set_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, pin_memory=True
    )


def _create_folder(path: pathlib.PosixPath) -> None:
    """Args:
    path (pathlib.Path): relative path to be created"""
    if isinstance(path, str):
        path = Path(path)

    try:
        Path.mkdir(path, parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def _get_root() -> str:
    project_root = Path(__file__).absolute().parent.parent.parent

    return project_root


def update_config_paths(config: dict):
    path = _get_root()
    keys = [
        key
        for key in config.keys()
        if key in ["data_path", "logs_path", "models_path", "results_path"]
    ]
    values = [path / config[key] for key in keys]
    dictionary = dict(zip(keys, values))
    config.update(dictionary, allow_val_change=True)

    config.device = get_device()
    _create_folder(config.data_path)
    _create_folder(config.models_path)
    _create_folder(config.results_path)
    _create_folder(config.logs_path)

    return config
