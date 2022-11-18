import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader, Subset

from .transforms import get_image_transform


class AbstractDataset(torchvision.datasets.VisionDataset):
    """Abstract dataset class for all datasets which are available in torchvision.datasets."""

    def __init__(self, name: str, data_path: str, image_size: int, train: bool):
        super().__init__(root=data_path, transform=None, target_transform=None)

        self.name = name
        self.data_path = data_path
        self.image_size = image_size
        self.train = train
        self.transform = get_image_transform(image_size=self.image_size)

        if self.name == "MNIST":
            self.dataset = torchvision.datasets.MNIST(
                root=self.data_path, train=self.train, download=True
            )
            # get unique classes from dataset (MNIST has 10 classes)
            # .numpy() is needed because the classes are in a torch tensor
            self.classes = list(np.unique(self.dataset.targets.numpy()))
        elif self.name == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.data_path, train=self.train, download=True
            )
            # get unique classes from dataset (CIFAR10 has 10 classes)
            # no .numpy() needed because the classes are in a numpy array
            self.classes = list(np.unique(self.dataset.targets))
        else:
            raise NotImplementedError(
                f"Dataset {self.name} is not implemented. Please choose from MNIST or CIFAR10"
            )

    def __getitem__(self, index):
        image, target = self.dataset[index]

        if self.transform is not None:
            image = self.transform(image)

        return image, target

    def __len__(self):
        return len(self.dataset)


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


def get_data(config: dict, testing: bool):
    """Creates train and val data loader and return also the classes and number
    of classes.
    """
    name = config.data["dataset"]
    image_size = config.data["image_size"]
    data_path = config.data_paths["data"]

    trainset = AbstractDataset(name, data_path, image_size, train=True)
    testset = AbstractDataset(name, data_path, image_size, train=False)
    classes = trainset.classes
    num_classes = len(classes)

    trainset, valset = _split_train_val(trainset, val_split=0.1)

    if testing:
        indices = np.arange(0, 20)
        trainset = Subset(trainset, indices)
        valset = Subset(valset, indices)
        testset = Subset(testset, indices)

    batch_size = config.batch_size
    train_loader = set_dataloader(trainset, batch_size)
    val_loader = set_dataloader(valset, batch_size)
    test_loader = set_dataloader(testset, batch_size)

    return train_loader, val_loader, test_loader, classes, num_classes
