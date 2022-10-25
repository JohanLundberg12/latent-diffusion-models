import os
import sys
import yaml
import wandb

import torch
import torch.nn.functional as f

from src import prepare_experiment
from src.utils import (
    make_settings,
    save_model,
)

from src.Autoencoder import Autoencoder
from src.AutoencoderTrainer import AutoencoderTrainer

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
# https://www.youtube.com/watch?v=9mS1fIYj1So
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
torch.backends.cudnn.benchmark = True


# reconstruction_term: how to tune?
def kl_loss_fn(data, outputs, mu, log_var, reconstruction_term=1):
    # reconstruction loss: element-wise mean squared error
    r_loss = f.mse_loss(
        input=outputs, target=data, reduction="mean"
    )  # sum or mean reduction?

    # Kullback Leibler Divergence loss
    # Ensures mu and sigma values never stray too far from a standard normal
    kl_loss = -0.5 * torch.sum(1 + log_var - mu**2 - torch.exp(log_var))

    # Evidence Lower Bound loss
    elbo_loss = reconstruction_term * r_loss + kl_loss

    return elbo_loss


def main(configurations):
    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config = prepare_experiment(config, delete=True)

        (train_loader, val_loader, classes, loss_fn, scaler) = make_settings(config)

        autoencoder = Autoencoder(
            in_channels=config.data["image_channels"],
            out_channels=config.data["image_channels"],
        )
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=config.learning_rate)

        wandb.watch(autoencoder, loss_fn, log="all", log_freq=10)
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("valid_loss", summary="min")

        trainer = AutoencoderTrainer(
            config=config,
            model=autoencoder,
            train_loader=train_loader,
            val_loader=val_loader,
            loss_fn=kl_loss_fn,
            optimizer=optimizer,
            scaler=scaler,
            classes=classes,
            device=config.device,
            epochs=config.epochs,
        )

        # Start train and eval steps
        trainer.train()

        save_model(
            model=autoencoder, target_dir="models", model_name=f"{config.name}" + ".pt"
        )
        wandb.save(os.path.join("models", f"{config.name}.pt"))


if __name__ == "__main__":
    # Get experiment configurations
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    main(configurations=configurations)
