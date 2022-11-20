from typing import Callable
from time import perf_counter
import errno
import importlib
import pathlib
from pathlib import Path
from tqdm import tqdm

import torch
import random
import numpy as np

from torchvision.utils import save_image


def create_folder(path: pathlib.PosixPath) -> None:
    """Creates a folder if it doesn't exist"""

    if isinstance(path, str):
        path = Path(path)

    try:
        Path.mkdir(path, parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def progress_bar(loader, desc=""):
    """Returns a progress bar for a given loader"""

    return tqdm(enumerate(loader), total=len(loader), desc=desc)


def load_model(model: torch.nn.Module, state_dict_path: str) -> torch.nn.Module:
    """Loads a model from a state dict path"""

    # load model weights
    state_dict = torch.load(state_dict_path)

    # load weights into model
    model.load_state_dict(state_dict)

    return model


def get_obj_from_str(string: str, reload=False) -> object:
    """Returns an object from a string"""
    # get_obj_from_str example:
    # string = "torch.nn.Conv2d"
    # module = torch.nn
    # cls = Conv2d

    # Split string into module and class
    module_name, cls_name = string.rsplit(".", 1)

    # Import module
    module = importlib.import_module(module_name)

    # reload module if True
    if reload:
        importlib.reload(module)

    cls = getattr(module, cls_name)

    return cls


def instantiate_from_config(config: dict) -> torch.nn.Module:
    """Instantiates a class from a config dict"""
    # instantiate_from_config example:
    # config = {
    #     "target": "torch.nn.Conv2d",
    #     "params": {
    #         "in_channels": 3,
    #         "out_channels": 64,
    #         "kernel_size": 3,
    #         "stride": 1,
    #         "padding": 1,
    #     },
    # }

    cls = get_obj_from_str(config["target"])

    params = config["params"]

    return cls(**params)


# state_dict_path example: 'load/from/path/model.pth'
def get_model_from_config(config, state_dict_path: str = None) -> torch.nn.Module:
    """Instantiates a model from a config dict"""

    # Get model
    model = instantiate_from_config(config)

    # Load state dict if path provided
    if state_dict_path is not None:
        print(f"Loading model from {state_dict_path}")

        model = load_model(model, state_dict_path)

    return model


def timeit(method: Callable) -> Callable:
    """Decorator to time a function"""

    def timed(*args, **kw):
        """Returns the time taken to execute a function"""
        ts = perf_counter()
        result = method(*args, **kw)
        te = perf_counter()
        print(f"{method.__name__}: {te - ts} sec")
        return result

    return timed


def save_images(imgs: torch.Tensor, name: str, ext: str = ".png"):
    """Saves each image in the tensor to name

    Args:
        imgs (torch.Tensor): (batches, channels, H, W)
        name (str): filename
        ext (str, optional): extension of images. Defaults to ".png".
    """
    for i in range(imgs.size(0)):
        save_image(imgs[i, :, :, :], "./{}_{}{}".format(name, i, ext))


def get_device():
    """Returns the device to be used for training."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} as backend")

    return device


def set_seed(seed: int = 42):
    """Sets the seed for the experiment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed
