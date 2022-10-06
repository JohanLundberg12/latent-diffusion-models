from pathlib import Path
from typing import Callable, Type
from time import time
import numpy as np
import torch
import torch.utils.data
import torch.optim as optim
from torch import nn
from tqdm import tqdm
from wandb.sdk.wandb_run import Run
from torch.utils.tensorboard import SummaryWriter

from models.Unet import UNet
from models.DDPM import Diffusion


class DiffusionModelTrainer:
    """Configurations"""

    def __init__(
        self,
        run: Type[Run],
        diffusion_model: Diffusion,
        eps_model: UNet,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        cfg_scale: int,  # classifier free guidance scale
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler,
    ) -> None:
        self.run = run
        self.device = run.config.device
        self.epochs = run.config.epochs

        self.diffusion_model = diffusion_model.to(self.device)
        self.eps_model = eps_model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg_scale = cfg_scale
        self.classes = torch.tensor(run.config.classes).to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler

        self.progress_bar = tqdm(range(self.epochs * len(self.train_loader)))
        self.writer = SummaryWriter(
            log_dir=str(self.run.config.results_path),
            filename_suffix=self.run.config.name,
        )

    def train(self):
        """
        ### Train
        """
        train_loss: float = 0.0
        self.eps_model.train()

        # Iterate through the dataset
        for _, (data, targets) in enumerate(self.train_loader):
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

            self.progress_bar.update(1)

            # Add batch loss to total loss
            train_loss += loss.item() * data.size(0)
        # Calculate average loss
        train_loss /= len(self.train_loader)

        self.run.log({"train_loss": train_loss})

        return train_loss

    def eval(self):
        valid_loss: float = 0.0

        self.eps_model.eval()

        with torch.no_grad():
            for _, (data, targets) in enumerate(self.val_loader):
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
            self.run.log({"val_loss": valid_loss})

        return valid_loss

    def run_training(self):
        """
        ### Training loop
        """
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
            self.writer.add_scalar("Loss/train", train_loss, epoch)
            self.writer.add_scalar("Loss/val", valid_loss, epoch)
            self.writer.close()

        torch.save(
            self.eps_model.state_dict(),
            Path(self.run.config.models_path) / f"model_{self.run.config.name}.pt",
        )

        return train_losses, valid_losses


class AutoEncoderTrainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
