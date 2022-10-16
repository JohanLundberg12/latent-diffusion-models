"""Script to train a pixel or latent diffusion model
"""
import os.path
import sys
import torch
import yaml
import wandb
from src import prepare_experiment

from src.utils import (
    instantiate_unet,
    load_model,
    make_settings,
    save_model,
)

from src.Autoencoder import Autoencoder
from src.DDPM import Diffusion
from src.DiffusionModelTrainer import DiffusionModelTrainer

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

        eps_model = instantiate_unet(config)
        optimizer = torch.optim.Adam(eps_model.parameters(), lr=config.learning_rate)

        if config.diffusion["type"] == "latent":
            autoencoder = Autoencoder()
            load_model(model=autoencoder, config=config, name="autoencoder")
        else:
            autoencoder = None

        # Create DDPM model
        diffusion_model = Diffusion(
            n_steps=config.diffusion["n_steps"],
            device=config.device,
        )

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
            autoencoder=autoencoder,
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
