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

# note to self:
# some arguments should be model invariant and some should be
# noise type dependent
# then we need to instantiate a particular diffusion model and
# noise type depending on diffusion_type and noise_type

if __name__ == "__main__":

    testing = True
    set_seeds()

    # Get experiment configurations
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    # new experiment run
    run = wandb.init(
        project=configurations["name"], entity="itu-gen", config=configurations
    )
    wandb.define_metric("train_loss", summary="min")
    wandb.define_metric("valid_loss", summary="min")
    config = run.config
    config.device = get_device()

    # Load data
    train_loader, val_loader, classes, num_classes = get_data(config, testing=testing)

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

    trainer = DiffusionModelTrainer(
        run=run,
        diffusion_model=diffusion_model,
        eps_model=eps_model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg_scale=3,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scaler=scaler,
    )

    # Start train and eval steps
    trainer.train()

    save_model(
        model=eps_model, target_dir="models", model_name=f"{config.name}" + ".pt"
    )
    wandb.save("model_" + str(config.name))
