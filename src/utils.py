import argparse
import errno
import pathlib
from pathlib import Path
from tqdm import tqdm

import torch
import torch.nn.functional as f

from .data_utils import get_data


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


def save_model(model: torch.nn.Module, target_dir: str, model_name: str):
    # Create dir if not exists
    target_dir_path = Path(target_dir)
    target_dir_path.mkdir(parents=True, exist_ok=True)

    assert model_name.endswith(".pth") or model_name.endswith(
        ".pt"
    ), "should end with '.pt' or '.pth'"

    # Create model save path
    model_save_path = target_dir_path / model_name

    # Save the model state_dict()
    print(f"[INFO] Saving model to: {model_save_path}")
    torch.save(obj=model.state_dict(), f=model_save_path)


def progress_bar(loader, desc=""):
    return tqdm(enumerate(loader), total=len(loader), desc=desc)


def load_model(model, path: str):
    model.load_state_dict(torch.load(path))


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


def _get_loss_fn(loss_fn):
    if loss_fn == "mse":
        return f.mse_loss
    elif loss_fn == "elbo":
        return elbo_loss_fn
    elif loss_fn == "cross-entropy":
        return f.cross_entropy
    else:
        raise NameError


def make_settings(config):
    """Args:
    config (dict): yaml configurations

    Returns:
        dict: Containing dataloaders, classes, loss_fn,
        scaler
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
