import argparse
import errno
import numpy as np
import pathlib
from pathlib import Path

import torch
import torchvision
from torch.utils.data import Subset

from .transforms import get_image_transform


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, config, image_size, train):
        transform = get_image_transform(image_size=image_size)

        super().__init__(
            str(config.data_path),
            train=train,
            download=True,
            transform=transform,
        )

    def __getitem__(self, item):
        return super().__getitem__(item)


def _get_dataset(config: dict, name: str, image_size: int, train):
    if name == "MNIST":
        dataset = MNISTDataset(config, image_size, train=train)

    return dataset


def _set_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, pin_memory=True
    )


def _get_classes(dataset):
    classes = list(set(dataset.targets.numpy()))

    return classes


def get_data(config: dict, testing: bool):
    """Creates train and val data loader and return also the classes and number
    of classes.
    """
    name = config["dataset"]
    image_size = config["image_size"]
    batch_size = config["batch_size"]

    trainset = _get_dataset(config, name, image_size, train=True)
    valset = _get_dataset(config, name, image_size, train=False)
    classes = _get_classes(trainset)

    if testing:
        indices = np.arange(0, 20)
        trainset = Subset(trainset, indices)
        valset = Subset(valset, indices)

    num_classes = len(classes)
    train_loader = _set_dataloader(trainset, batch_size)
    val_loader = _set_dataloader(valset, batch_size)

    return train_loader, val_loader, classes, num_classes


def create_folder(path: pathlib.PosixPath) -> None:
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
    """Returns the root of the project."""
    project_root = Path(__file__).absolute().parent.parent.parent

    return project_root


def create_parser():
    parser = argparse.ArgumentParser()

    # experiment specific args
    parser.add_argument("-ds", "--dataset", type=str, help="Training data")
    parser.add_argument(
        "-dft",
        "--diffusion_type",
        default="pixel",
        type=str,
        help="This is the type of space the forward diffusion happens at, latent or pixel space",
    )
    parser.add_argument(
        "-nt",
        "--noise_type",
        default="gaussian",
        type=str,
        help="This is the type of noise applied in the forward diffusion process",
    )

    # universal experiment settings
    parser.add_argument(
        "-t",
        "--time_steps",
        default=10,
        type=int,
        help="This is the number of steps T during the forward process",
    )
    parser.add_argument("-b", "--batch_size", default=2, type=int)
    parser.add_argument("-e", "--epochs", default=2, type=int)

    return parser


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} as backend")
    return device


def set_seeds(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    # Create dir if not exists
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    # Create model save path
    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "should end with '.pt' or '.pth'"

    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


# def update_config_paths(config: dict):
# """Changes the value of the keys to be the full path instead of
# just the folder name."""
# path = _get_root()
# keys = [
# key
# for key in config.keys()
# if key
# in ["data_path", "logs_path", "models_path", "results_path", "checkpoints_path"]
# ]
# values = [path / config[key] for key in keys]
# dictionary = dict(zip(keys, values))
# config.update(dictionary, allow_val_change=True)

# return config
