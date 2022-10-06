import torch
from pathlib import Path

from .Unet import UNet


def instantiate_model(config: dict, load=False):
    model = UNet(
        image_channels=config["image_channels"],
        n_channels=config["n_channels"],
        ch_mults=config["channel_multipliers"],
        out_channels=config["out_channels"],
        with_time_emb=config["with_time_emb"],
        num_classes=config["num_classes"],
    )
    if load:
        if isinstance(config.models_path, str):
            config.models_path = Path(config.models_path)
        model = model.load_state_dict(torch.load(config.models_path / config.name))

    return model
