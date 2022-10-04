import torch

from utils.data_utils import Experiment
from .Unet import UNet


def instantiate_model(experiment: Experiment, load=False):
    if load:
        model = torch.load(experiment.get_models_path / experiment.name)
    else:
        model = UNet(
            image_channels=experiment.config["image_channels"],
            n_channels=experiment.config["n_channels"],
            ch_mults=experiment.config["channel_multipliers"],
            out_channels=experiment.config["out_channels"],
            with_time_emb=experiment.config["with_time_emb"],
            num_classes=experiment.config["num_classes"],
        )

    return model
