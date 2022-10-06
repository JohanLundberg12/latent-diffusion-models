import sys
import numpy as np
import torch
import torch.nn.functional as f
import yaml
import wandb

from utils.data_utils import update_config_paths, get_dataset, set_dataloader

from models.DDPM import Diffusion
from models.utils import instantiate_model
from models.trainers import DiffusionModelTrainer

# note to self:
# some arguments should be model invariant and some should be
# noise type dependent
# then we need to instantiate a particular diffusion model and
# noise type depending on diffusion_type and noise_type


if __name__ == "__main__":
    testing = True

    # Get config experiment configurations
    config_file = sys.argv[1]
    config_from_file = yaml.safe_load(open(config_file, "r"))
    # _config = update_config(_config)

    # new experiment run
    run = wandb.init(
        project=config_from_file["name"], entity="itu-gen", config=config_from_file
    )
    config = run.config
    config.update(update_config_paths(config))

    # set seeds
    config.seed = 42
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data
    trainset, valset, classes = get_dataset(
        config, config["dataset"], config["image_size"], testing=testing
    )

    config.classes = classes
    num_classes = len(classes)
    config.num_classes = num_classes

    train_loader = set_dataloader(trainset, config["batch_size"])
    val_loader = set_dataloader(valset, config["batch_size"])

    # Create epsilon noise predictor
    eps_model = instantiate_model(config=config)

    # Create [DDPM class]
    diffusion_model = Diffusion(
        n_steps=config.n_steps,
        device=config.device,
    )
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
    trainer.run_training()
