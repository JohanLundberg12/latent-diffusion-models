import os.path
import sys
import torch
import torch.nn.functional as f
import yaml
import wandb

from src.utils.utils import (
    get_data,
    get_device,
)
from src.utils.utils import save_model, set_seeds

from src.modules.DDPM import Diffusion
from src.modules.utils import get_model
from src.modules.trainers import DiffusionModelTrainer
from src.modules.Sampler import Sampler

# note to self:
# some arguments should be model invariant and some should be
# noise type dependent
# then we need to instantiate a particular diffusion model and
# noise type depending on diffusion_type and noise_type


def make(config):

    # Load data
    train_loader, val_loader, classes, num_classes = get_data(
        config, testing=config.testing
    )
    # Update config
    config.classes = classes
    config.num_classes = num_classes

    # Create epsilon model
    eps_model = get_model(config=config)

    # Create DDPM model
    diffusion_model = Diffusion(
        n_steps=config.n_steps,
        device=config.device,
    )

    # Set scaler, optimizer and loss fn
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=config["learning_rate"])
    loss_fn = f.mse_loss

    return (
        eps_model,
        diffusion_model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        scaler,
        classes,
    )


def model_pipeline(configurations):
    with wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    ):
        config = wandb.config
        config.device = get_device()
        config = set_seeds(config)

        (
            eps_model,
            diffusion_model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scaler,
            classes,
        ) = make(config)
        wandb.watch(eps_model, criterion, log="all", log_freq=10)
        wandb.define_metric("train_loss", summary="min")
        wandb.define_metric("valid_loss", summary="min")

        trainer = DiffusionModelTrainer(
            config=config,
            diffusion_model=diffusion_model,
            eps_model=eps_model,
            train_loader=train_loader,
            val_loader=val_loader,
            cfg_scale=3,
            loss_fn=criterion,
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
        wandb.save(os.path.join("models", "first_experiment.pt"))

        # sampler = Sampler(config.name, config, diffusion_model, eps_model, classes)
        # sampler.sample(img_size=32)


if __name__ == "__main__":

    # Get experiment configurations
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    model_pipeline(configurations=configurations)
