import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader, Subset

from .transforms import get_image_transform


class CelebADataset(torchvision.datasets.CelebA):
    def __init__(self, config: dict, image_size: int, split: str) -> None:
        transform = get_image_transform(image_size=image_size)

        super().__init__(
            str(config.data_paths["data"]),
            split=split,
            download=True,
            transform=transform,
        )

    def __getitem__(self, item):
        return super().__getitem__(item)


class MNISTDataset(torchvision.datasets.MNIST):
    def __init__(self, config, image_size, train):
        transform = get_image_transform(image_size=image_size)

        super().__init__(
            str(config.data_paths["data"]),
            train=train,
            download=True,
            transform=transform,
        )

    def __getitem__(self, item):
        return super().__getitem__(item)


def _get_dataset(config: dict, name: str, image_size: int, train: bool):
    """Returns the dataset."""

    if name == "MNIST":
        dataset = MNISTDataset(config, image_size, train=train)
    elif name == "CelebA":
        if train:
            dataset = CelebADataset(config, image_size, split="train")
        else:
            dataset = CelebADataset(config, image_size, split="valid")

    else:
        raise NameError

    return dataset


def set_dataloader(dataset, batch_size):
    """Sets the data loader for the dataset."""
    # Use num_workers > 0 to enable asynchronous data processing.
    # pin_memory=True to make CPU to GPU copies asynchronous.
    # num_workers is usually to be tuned, depending on your hardware.

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)


def _get_classes(dataset) -> list:
    """Returns the classes of the dataset."""

    return list(set(dataset.targets.numpy()))


def _split_train_val(dataset, val_split: float) -> tuple:
    """Splits the dataset into train and validation set."""

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size

    return torch.utils.data.random_split(dataset, [train_size, val_size])


def get_data(config: dict, testing: bool):
    """Creates train and val data loader and return also the classes and number
    of classes.
    """
    name = config.data["dataset"]
    image_size = config.data["image_size"]
    batch_size = config.batch_size

    trainset = _get_dataset(config, name, image_size, train=True)
    classes = _get_classes(trainset)
    trainset, valset = _split_train_val(trainset, val_split=0.1)
    testset = _get_dataset(config, name, image_size, train=False)

    if testing:
        indices = np.arange(0, 20)
        trainset = Subset(trainset, indices)
        valset = Subset(valset, indices)
        testset = Subset(testset, indices)

    num_classes = len(classes)
    train_loader = set_dataloader(trainset, batch_size)
    val_loader = set_dataloader(valset, batch_size)
    test_loader = set_dataloader(testset, batch_size)

    return train_loader, val_loader, test_loader, classes, num_classes
