import sys

import torch
import yaml
from torchvision.datasets import ImageFolder

import wandb
from src import prepare_experiment
from src.data_utils import set_dataloader
from src.ResNetTrainer import ResNetTrainer
from src.transforms import get_gray_scale_image_transform
from src.utils import get_model_from_config, make_settings

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
torch.backends.cudnn.benchmark = True


def main(config: dict):

    (train_loader, val_loader, test_loader, classes, loss_fn, scaler) = make_settings(
        config
    )
    synthetic_data = load_synthetic_data(
        config.synthetic_data["path"], config.data["image_size"]
    )
    synthetic_dataloader = set_dataloader(
        synthetic_data,
        batch_size=config["batch_size"],
    )

    model = get_model_from_config(config["model"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    wandb.watch(model, criterion=loss_fn, log="all", log_freq=10)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("valid_loss", summary="min")
    wandb.define_metric("train_f1", summary="max")
    wandb.define_metric("valid_f1", summary="max")

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

    pretrain_loss, pretrain_f1 = trainer.pretrain(synthetic_dataloader)

    wandb.log({"pretrain_loss": pretrain_loss, "pretrain_f1": pretrain_f1})

    trainer.train()

    scores, avg_f1 = trainer.predict(test_loader)

    wandb.log({"avg_test_f1": avg_f1})
    wandb.log({"test_scores": scores})


def load_synthetic_data(path, image_size):
    transform = get_gray_scale_image_transform(image_size)

    # dataset containing (image, label)
    dataset = ImageFolder(path, transform=transform)

    return dataset


if __name__ == "__main__":
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config = prepare_experiment(config, delete=False)
        main(config=config)
