import errno
import importlib
import pathlib
from pathlib import Path
from typing import Callable
from tqdm import tqdm

import torch
import torch.nn.functional as f

from .data_utils import get_data


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
    weights = torch.load(state_dict_path)

    # load weights into model
    model.load_state_dict(weights)

    return model


# reconstruction_term: how to tune?
def elbo_loss_fn(data, outputs, mu, log_var, reconstruction_term=1):
    # reconstruction loss: element-wise mean squared error
    r_loss = f.mse_loss(
        input=outputs, target=data, reduction="mean"
    )  # sum or mean reduction (not important right now)?

    # Kullback Leibler Divergence loss
    # Ensures mu and sigma values never stray too far from a standard normal
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

    # Evidence Lower Bound loss
    elbo_loss = reconstruction_term * r_loss + kl_loss

    return elbo_loss


def _get_loss_fn(loss_fn) -> Callable:
    """Returns a loss function"""

    if loss_fn == "mse":
        return f.mse_loss
    elif loss_fn == "elbo":
        return elbo_loss_fn
    elif loss_fn == "cross-entropy":
        return f.cross_entropy
    else:
        raise NameError


def make_settings(config):
    """Creates a settings dict from a config dict
    Args:
        config (dict): yaml configurations
    Returns:
        dict: Containing dataloaders, classes, loss_fn,
        scaler: GradScaler for mixed precision training
    """
    # Load data
    train_loader, val_loader, classes, num_classes = get_data(
        config, testing=config.testing
    )
    # Update config
    config.classes = classes
    config.num_classes = num_classes

    # get loss fn (criterion)
    loss_fn = _get_loss_fn(config.loss_fn)

    # instances of GradScaler() help perform steps of the gradient scaling
    # conveniently. Improves convergence for networks with
    # float16 (Conv layers) gradients by minimizing gradient underflow
    scaler = torch.cuda.amp.GradScaler()

    return (train_loader, val_loader, classes, loss_fn, scaler)


def get_obj_from_str(string: str, reload=False):
    """This function takes a string and returns the object
    Args:
        string (_type_): path to the object
        reload (bool, optional): Whether to reload module. Defaults to False.
    Returns:
        object: The object
    """
    # Split string into module and class
    module, cls = string.rsplit(".", 1)

    # Import module
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)

    # Get class
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: dict) -> torch.nn.Module:
    """Instantiates a class from a config dict"""

    # Get class
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


# state_dict_path example: 'load/from/path/model.pth'
def get_model_from_config(config, state_dict_path: str = None) -> torch.nn.Module:
    """Instantiates a model from a config dict"""

    # Get model
    model = instantiate_from_config(config["model"])

    # Load state dict if path provided
    if state_dict_path:
        print(f"Loading model from {state_dict_path}")

        model = load_model(model, state_dict_path)

    return model
