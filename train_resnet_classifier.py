import sys
import os

import torch
import yaml
from torchvision.datasets import ImageFolder

import wandb
from src.Config import Config
from src.data_utils import create_dataloaders, set_dataloader
from src.ResNetTrainer import ResNetTrainer
from src.transforms import get_gray_scale_image_transform
from src.utils import get_model_from_config

# cudnn autotuner is going run a short benchmark and will select the algorithm
# with the best performance on a given hardware for a given input size.
# Adds small overhead during the first training iteration,
# but can lead to a significant speed up for conv layers.
torch.backends.cudnn.benchmark = True

os.environ["WANDB_MODE"] = "dryrun"


def load_synthetic_data(path, image_size):
    transform = get_gray_scale_image_transform(image_size)

    # dataset containing (image, label)
    dataset = ImageFolder(path, transform=transform)

    return dataset


def main(config: dict):

    (train_loader, val_loader, test_loader, classes) = create_dataloaders(config)

    if config.pretrain:
        synthetic_data = load_synthetic_data(
            config.synthetic_data["path"], config.data["image_size"]
        )
        synthetic_dataloader = set_dataloader(
            synthetic_data,
            batch_size=config["batch_size"],
        )

    model = get_model_from_config(config["model"])

    trainer = ResNetTrainer(config, model, train_loader, val_loader, classes)

    wandb.watch(model, criterion=trainer.loss_fn, log="all", log_freq=10)
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("valid_loss", summary="min")
    wandb.define_metric("train_f1", summary="max")
    wandb.define_metric("valid_f1", summary="max")

    if config.pretrain:
        pretrain_loss, pretrain_f1 = trainer.pretrain(synthetic_dataloader)

        wandb.log({"pretrain_loss": pretrain_loss, "pretrain_f1": pretrain_f1})

    trainer.train()

    scores, avg_f1 = trainer.predict(test_loader)

    wandb.log({"avg_test_f1": avg_f1})
    wandb.log({"test_scores": scores})


if __name__ == "__main__":
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    config = Config(**configurations)

    with wandb.init(project=config.project_name, entity=config.entity, config=config):
        main(wandb.config)
