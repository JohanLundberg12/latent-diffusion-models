from typing import Callable
from time import time
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from tqdm import tqdm
from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
from utils.data_utils import Experiment

from models.Unet import UNet
from models.DDPM import Diffusion


@dataclass
class Trainer:
    """Configurations"""

    name: str
    experiment: Experiment
    device: torch.device
    eps_model: UNet
    epochs: int  # 100
    scaler: torch.cuda.amp.grad_scaler
    optimizer: optim.Optimizer
    learning_rate: float  # 2e - 5

    diffusion_model: Diffusion
    n_steps: int  # T: 100-400
    # Number of samples to generate
    n_samples: int  # generated images 16

    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    image_size: int
    image_channels: int  # 1 or 3
    n_channels: int  # Number of output channels in the first feature map
    # The number of channels is `channel_multipliers[i] * n_channels` -> [64, 128, 256, 512]
    channel_multipliers: list[int]

    classes: list()
    cfg_scale: int
    progress_bar: None
    loss_fn: Callable

    def train(self):
        """
        ### Train
        """
        train_loss: float = 0.0
        self.eps_model.train()

        # Iterate through the dataset
        for batch_idx, (data, targets) in enumerate(self.train_loader):
            # Move data to device
            data = data.to(self.device)
            targets = targets.to(self.device)

            with torch.cuda.amp.autocast():
                noise, xt, t = self.diffusion_model(data)

                if np.random.random() < 0.1:
                    targets = None

                # Make the gradients zero
                self.optimizer.zero_grad()

                eps_theta = self.eps_model(xt, t, targets)

                # calculate loss
                loss = self.loss_fn(noise, eps_theta)

            # Scale gradients
            self.scaler.scale(loss).backward()
            # Update optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Compute gradients
            # loss.backward()
            # Take an optimization step
            # self.optimizer.step()
            self.progress_bar.update(1)

            # Add batch loss to total loss
            train_loss += loss.item() * data.size(0)
        # Calculate average loss
        train_loss /= len(self.train_loader)

        return train_loss

    def eval(self):
        valid_loss: float = 0.0

        self.eps_model.eval()

        with torch.no_grad():
            for batch_idx, (data, targets) in enumerate(self.val_loader):
                data = data.to(self.device)
                targets = targets.to(self.device)

                noise, xt, t = self.diffusion_model(data)
                eps_theta = self.eps_model(xt, t, targets)
                if self.cfg_scale > 0:
                    eps_theta_uncond = self.eps_model(xt, t, None)
                    eps_theta = torch.lerp(eps_theta_uncond, eps_theta, self.cfg_scale)
                loss = self.loss_fn(noise, eps_theta)
                valid_loss += loss.item() * data.size(0)

            # Calculate average loss
            valid_loss /= len(self.val_loader)

        return valid_loss

    def run(self):
        """
        ### Training loop
        """
        self.diffusion_model.to(device=self.device)
        self.eps_model.to(device=self.device)

        self.progress_bar = tqdm(range(self.epochs * len(self.train_loader)))

        writer = SummaryWriter(log_dir=str(self.experiment.path) + "/results")

        self.classes = torch.tensor(self.classes).to(self.device)

        train_losses = list()
        valid_losses = list()

        for epoch in range(self.epochs):
            start = time()
            train_loss = self.train()
            valid_loss = self.eval()
            stop = time()

            print(
                f"\nEpoch: {epoch}",
                f"\navg train-loss: {round(train_loss, 4)}",
                f"\navg val-loss: {round(valid_loss, 4)}",
                f"\ntime: {stop-start}\n",
            )

            # Save losses
            train_losses.append(train_loss),
            valid_losses.append(valid_loss)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", valid_loss, epoch)
            writer.close()

            # Sample some images
            self.sample()

        torch.save(
            self.eps_model.state_dict(),
            str(self.experiment.models_path) + "model.pt",
        )
