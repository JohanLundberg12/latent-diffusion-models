"""Script to train a pixel or latent diffusion model
"""
import sys
import os

import torch
import yaml

import wandb
from src.Config import Config
from src.DiffusionModelTrainer import DiffusionModelTrainer
from src.data_utils import create_dataloaders
from src.utils import get_model_from_config

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
torch.backends.cudnn.benchmark = True

os.environ["WANDB_MODE"] = "dryrun"


def main(config: dict):

    (train_loader, val_loader, classes) = create_dataloaders(config)

    # Load model
    model = get_model_from_config(config["model"])
    diffusion = get_model_from_config(config["diffusion"])
    cfg_scale = config["diffusion"]["cfg_scale"]

    trainer = DiffusionModelTrainer(
        config, model, train_loader, val_loader, classes, diffusion, cfg_scale
    )

    wandb.watch(model, trainer.loss_fn, log="all", log_freq=10)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("valid_loss", summary="min")

    # Start train and eval steps
    trainer.train()


if __name__ == "__main__":
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    config = Config(**configurations)

    with wandb.init(
        project=config.project_name,
        entity=config.entity,
        config=config,
    ):
        main(wandb.config)
