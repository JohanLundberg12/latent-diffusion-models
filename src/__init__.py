from pathlib import Path
import numpy as np
import shutil
from torchvision.utils import save_image
import torch

from src.utils import create_folder


def _get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} as backend")

    return device


def _set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    return seed


def _setup_experiment_dirpath(config: dict, delete=False) -> Path:
    """Creates a folder for the experiment with a subfolder 'images.
    If the folder already exists, it is either deleted or a version 2
    of that folder is created.

    Args:
        config (dict): Is a Wandb config object
    """
    directory = config.type
    name = config.name

    dirpath = Path(directory) / name
    if dirpath.exists() and dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)

    return dirpath


def prepare_experiment(config: dict, delete=False):
    config.seed = _set_seed()
    config.device = _get_device()
    dirpath = _setup_experiment_dirpath(config)
    config.data_paths = {"dirpath": str(dirpath)}
    config.data_paths["data"] = "data"

    for path in ["results", "models", "logs", "checkpoints"]:
        config.data_paths[path] = str(dirpath / path)
        create_folder(dirpath / path)

    return config


def save_images(imgs: torch.Tensor, name: str, ext: str = ".png"):
    """Saves each image in the tensor to name

    Args:
        imgs (torch.Tensor): (batches, channels, H, W)
        name (str): filename
        ext (str, optional): extension of images. Defaults to ".png".
    """
    for i in range(imgs.size(0)):
        save_image(imgs[i, :, :, :], "./{}_{}{}".format(name, i, ext))
