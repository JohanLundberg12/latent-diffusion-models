import sys
import numpy as np
import torch
import torch.nn.functional as f
from torch.utils.data import Subset

from utils.data_utils import Experiment, get_dataset, set_dataloader

from models.DDPM import Diffusion
from models.utils import instantiate_model
from models.Trainer import Trainer

# note to self:
# some arguments should be model invariant and some should be
# noise type dependent
# then we need to instantiate a particular diffusion model and
# noise type depending on diffusion_type and noise_type


def main(config, experiment, trainset, valset, classes):

    num_classes = len(classes)
    config["num_classes"] = num_classes

    train_loader = set_dataloader(trainset, config["batch_size"])
    val_loader = set_dataloader(valset, config["batch_size"])

    # Create epsilon noise predictor
    eps_model = instantiate_model(experiment=experiment)

    device = experiment.device
    # Create [DDPM class]
    diffusion_model = Diffusion(
        n_steps=config["n_steps"],
        device=device,
    )
    learning_rate = config["learning_rate"]
    scaler = torch.cuda.amp.GradScaler()
    optimizer = torch.optim.Adam(eps_model.parameters(), lr=learning_rate)

    trainer = Trainer(
        name=config["name"],
        experiment=experiment,
        device=device,
        eps_model=eps_model,
        epochs=config["epochs"],
        scaler=scaler,
        optimizer=optimizer,
        learning_rate=config["learning_rate"],
        diffusion_model=diffusion_model,
        n_steps=config["n_steps"],
        n_samples=config["n_samples"],
        train_loader=train_loader,
        val_loader=val_loader,
        image_size=config["image_size"],
        image_channels=config["image_channels"],
        channel_multipliers=config["channel_multipliers"],
        classes=classes,
        cfg_scale=3,
        n_channels=config["n_channels"],
        progress_bar=None,
        loss_fn=f.mse_loss,
    )
    trainer.run()


if __name__ == "__main__":
    testing = True

    # Create experiment
    experiment = Experiment()

    # Get config experiment configurations
    config_file = sys.argv[1]
    config = experiment.load_config_file(config_file)

    # Load data
    trainset, valset = get_dataset(experiment, config["dataset"], config["image_size"])
    classes = list(set(trainset.targets.numpy()))
    if testing:
        indices = np.arange(0, 100)
        trainset = Subset(trainset, indices)
        valset = Subset(valset, indices)

    main(
        config=config,
        experiment=experiment,
        trainset=trainset,
        valset=valset,
        classes=classes,
    )
