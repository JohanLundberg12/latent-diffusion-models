import torch
import torchvision
import numpy as np

from .transforms import get_image_transform


class AbstractDataset(torchvision.datasets.VisionDataset):
    """Abstract dataset class for all datasets which are available in torchvision.datasets."""

    def __init__(
        self,
        name: str,
        data_path: str,
        image_size: int,
        train: bool,
        debugging: bool = False,
    ):
        super().__init__(root=data_path, transform=None, target_transform=None)

        self.name = name
        self.data_path = data_path
        self.image_size = image_size
        self.train = train
        self.debugging = debugging
        self.transform = get_image_transform(self.image_size)

        if self.name == "MNIST":
            self.dataset = torchvision.datasets.MNIST(
                root=self.data_path,
                train=self.train,
                download=True,
                transform=self.transform,
            )

            # get unique classes from dataset (MNIST has 10 classes)
            # .numpy() is needed because the classes are in a torch tensor
            self.classes = list(np.unique(self.dataset.targets.numpy()))
        elif self.name == "CIFAR10":
            self.dataset = torchvision.datasets.CIFAR10(
                root=self.data_path,
                train=self.train,
                download=True,
                transform=self.transform,
            )
            # get unique classes from dataset (CIFAR10 has 10 classes)
            # no .numpy() needed because the classes are in a numpy array
            self.classes = list(np.unique(self.dataset.targets))
        else:
            raise NotImplementedError(
                f"Dataset {self.name} is not implemented. Please choose from MNIST or CIFAR10"
            )
        if self.debugging:
            indices = np.arange(0, 20)
            self.dataset = torch.utils.data.Subset(self.dataset, indices)

    def __getitem__(self, index: int) -> tuple:
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)
