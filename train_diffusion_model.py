"""Script to train a pixel or latent diffusion model
"""
import os.path
import sys
import torch
import yaml
import wandb

from src.Autoencoder import Autoencoder
from src.DDPM import Diffusion
from src.DiffusionModelTrainer import DiffusionModelTrainer
from src.LatentDiffusionModel import LatentDiffusionModel
from src.UNet import UNet

from src import prepare_experiment
from src.utils import (
    load_model,
    make_settings,
    save_model,
)


# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
torch.backends.cudnn.benchmark = True

# note to self:
# some arguments should be model invariant and some should be
# noise type dependent
# then we need to instantiate a particular diffusion model and
# noise type depending on diffusion_type and noise_type


def main(configurations):
    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config = prepare_experiment(config, delete=True)

        (train_loader, val_loader, classes, loss_fn, scaler) = make_settings(config)

        # Create DDPM model
        diffusion_model = Diffusion(
            n_steps=config.diffusion["n_steps"],
            device=config.device,
        )

        if config.diffusion["type"] == "latent":
            autoencoder = Autoencoder(
                in_channels=config.data["image_channels"],
                out_channels=config.data["image_channels"],
            )
            load_model(autoencoder, path="models/" + config.autoencoder["name"])

            eps_model = UNet(
                in_channels=autoencoder.z_channels,
                out_channels=autoencoder.z_channels,
                channels=config.model["params"]["channels"],
                channel_multipliers=config.model["params"]["channel_multipliers"],
                with_time_emb=config.model["params"]["with_time_emb"],
                num_classes=config.num_classes,
            )
            eps_model = LatentDiffusionModel(
                eps_model=eps_model,
                autoencoder=autoencoder,
                latent_scaling_factor=1,
                n_steps=config.diffusion["n_steps"],
                linear_start=0.05,
                linear_end=0.1,
            )
        else:
            eps_model = UNet(
                in_channels=config.data["image_channels"],
                out_channels=config.model["params"]["out_channels"],
                channels=config.model["params"]["channels"],
                channel_multipliers=config.model["params"]["channel_multipliers"],
                with_time_emb=config.model["params"]["with_time_emb"],
                num_classes=config.num_classes,
            )
        optimizer = torch.optim.Adam(eps_model.parameters(), lr=config.learning_rate)

        wandb.watch(eps_model, loss_fn, log="all", log_freq=10)
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("valid_loss", summary="min")

        trainer = DiffusionModelTrainer(
            config=config,
            diffusion_model=diffusion_model,
            eps_model=eps_model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg_scale=3,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            classes=classes,
            device=config.device,
            epochs=config.epochs,
        )

        # Start train and eval steps
        trainer.train()

        save_model(
            model=eps_model, target_dir="models", model_name=f"{config.name}" + ".pt"
        )
        wandb.save(os.path.join("models", f"{config.name}.pt"))


if __name__ == "__main__":
    # Get experiment configurations
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    main(configurations=configurations)
