import torch
from torch.utils.data import DataLoader

from .AbstractDataset import AbstractDataset


def set_dataloader(dataset, batch_size):
    """Sets the data loader for the dataset."""
    # Use num_workers > 0 to enable asynchronous data processing.
    # pin_memory=True to make CPU to GPU copies asynchronous.
    # num_workers is usually to be tuned, depending on your hardware.

    return DataLoader(dataset, batch_size, shuffle=True, num_workers=4, pin_memory=True)


def _split_train_val(dataset, val_split: float) -> tuple:
    """Splits the dataset into train and validation set."""

    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size

    return torch.utils.data.random_split(dataset, [train_size, val_size])


# create data loaders
def create_dataloaders(config: dict) -> tuple:
    """Creates the train, validation and test dataloaders."""
    # either return the train, validation and test dataloaders or
    # train and test dataloaders if validation is not required.

    name = config.data["dataset"]
    image_size = config.data["image_size"]
    batch_size = config.batch_size
    debugging = config.debugging

    dataset = AbstractDataset(
        name, data_path="data", image_size=image_size, train=True, debugging=debugging
    )
    classes = dataset.classes

    testset = AbstractDataset(
        name, data_path="data", image_size=image_size, train=False, debugging=debugging
    )
    test_loader = set_dataloader(testset, batch_size)

    if config["data"]["val_split"] > 0:
        val_split = config["data"]["val_split"]
        trainset, valset = _split_train_val(dataset, val_split=val_split)
        train_loader = set_dataloader(trainset, batch_size)
        val_loader = set_dataloader(valset, batch_size)

        return train_loader, val_loader, test_loader, classes
    else:
        train_loader = set_dataloader(dataset, batch_size)

        return train_loader, test_loader, classes
