import argparse
import errno
import pathlib
from pathlib import Path
from tqdm import tqdm

import torch
from torch import nn
import torch.nn.functional as f

from .UNet import UNet

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


def load_model(model, config, name):
    model.load_state_dict(torch.load(config.models_path + "/" + name + ".pt"))


def instantiate_unet(config: dict = None):
    model = UNet(
        image_channels=config.data["image_channels"],
        n_channels=config.model["params"]["n_channels"],
        ch_mults=config.model["params"]["channel_multipliers"],
        out_channels=config.model["params"]["out_channels"],
        with_time_emb=config.model["params"]["with_time_emb"],
        num_classes=config.num_classes,
    )

    return model


def _get_loss_fn(loss_fn):
    if loss_fn == "mse":
        return f.mse_loss
    elif loss_fn == "KL":
        return nn.KLDivLoss(reduction="batchmean")


def make_settings(config):
    """Args:
    config (dict): yaml configurations

    Returns:
        dict: Containing dataloaders, classes, criterion,
        scaler
    """
    # Load data
    train_loader, val_loader, classes, num_classes = get_data(
        config, testing=config.testing
    )
    # Update config
    config.classes = classes
    config.num_classes = num_classes

    # Set scaler, optimizer and loss fn (criterion)
    loss_fn = _get_loss_fn(config.loss_fn)
    scaler = torch.cuda.amp.GradScaler()

    return (train_loader, val_loader, classes, loss_fn, scaler)
