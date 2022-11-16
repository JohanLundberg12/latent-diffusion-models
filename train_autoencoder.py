import sys

import torch
import yaml

import wandb
from src import prepare_experiment
from src.AutoencoderTrainer import AutoencoderTrainer
from src.utils import get_model_from_config, make_settings

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
# https://www.youtube.com/watch?v=9mS1fIYj1So
# https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
torch.backends.cudnn.benchmark = True


def main(configurations):
    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config = prepare_experiment(config, delete=True)

        (train_loader, val_loader, classes, loss_fn, scaler) = make_settings(config)

        model = get_model_from_config(config)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        wandb.watch(model, loss_fn, log="all", log_freq=10)
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("valid_loss", summary="min")

        trainer = AutoencoderTrainer(
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

        # Start train and eval steps
        trainer.train()


if __name__ == "__main__":
    # Get experiment configurations
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    main(configurations=configurations)
