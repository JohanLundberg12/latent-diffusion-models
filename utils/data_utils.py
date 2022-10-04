import torch
import torchvision

from .experiment import Experiment
from .transforms import get_image_transform


class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, experiment, image_size, train):
        self.train = train
        transform = get_image_transform(image_size=image_size)

        super().__init__(
            str(experiment.get_data_path()),
            train=self.train,
            download=True,
            transform=transform,
        )

    def __getitem__(self, item):
        return super().__getitem__(item)


def get_dataset(experiment: Experiment, name: str, image_size: int):
    if name == "MNIST":
        return MNISTDataset(experiment, image_size, train=True), MNISTDataset(
            experiment, image_size, train=False
        )


def set_dataloader(dataset, batch_size):
    return torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=True, pin_memory=True
    )


# def get_paths(
# root_path: pathlib.PosixPath, dataset: str, diffusion_type: str, noise_type: str
# ) -> List[pathlib.PosixPath]:

# path = root_path / Path(f"data/{dataset}/{diffusion_type}/{noise_type}")
# create_folder(path / "results")
# create_folder(path / "models")

# data_path = root_path / "data" / f"{dataset}"
# model_path = path / "models"
# results_path = path / "results"

# return data_path, str(path), model_path, results_path
