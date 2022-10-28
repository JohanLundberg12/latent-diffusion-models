import os
import sys
import yaml
import wandb

import torch

from src.ResNetClassifier import ResNetBase
from src.ResNetTrainer import ResNetTrainer

from src import prepare_experiment
from src.utils import make_settings, save_model

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
torch.backends.cudnn.benchmark = True


def main(config: dict):

    (train_loader, val_loader, classes, loss_fn, scaler) = make_settings(config)

    # n_blocks[0] = 2 means two resnet blocks for the feature mapping with 64 channels
    # also, this one won't have any downscaling as this was done in the initial conv layer.
    model = ResNetBase(
        out_channels=len(classes),
        n_blocks=[2, 2, 2, 2],
        n_channels=[64, 128, 256, 512],
        img_channels=config.data["image_channels"],
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, loss_fn, log="all", log_freq=10)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("valid_loss", summary="min")

    trainer = ResNetTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
        classes=classes,
        device=config.device,
        epochs=config.epochs,
    )

    trainer.train()

    save_model(model=model, target_dir="models", model_name=f"{config.name}" + ".pt")
    wandb.save(os.path.join("models", f"{config.name}.pt"))


if __name__ == "__main__":
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config = prepare_experiment(config, delete=True)
        main(config=config)
