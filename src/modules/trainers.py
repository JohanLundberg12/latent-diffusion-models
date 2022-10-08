from typing import Callable, Type
from time import time
import numpy as np

import torch
from torch import nn
import torch.utils.data
import torch.optim as optim

from tqdm import tqdm
from wandb.sdk.wandb_run import Run

from .Unet import UNet
from .DDPM import Diffusion
from .utils import save_checkpoint


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
        self.device = self.run.config.device
        self.epochs = self.run.config.epochs

        self.diffusion_model = diffusion_model.to(self.device)
        self.eps_model = eps_model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg_scale = cfg_scale  # classifier free guidance
        self.classes = torch.tensor(run.config.classes).to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler

    def train_step(self):
        """
        ### Train
        """
        self.eps_model.train()

        train_loss: float = 0.0

        pbar = tqdm(
            enumerate(self.train_loader),
            total=len(self.train_loader),
            desc="train step",
        )
        start_time = time()

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            with torch.cuda.amp.autocast():
                # diffusion model forward pass
                noise, xt, t = self.diffusion_model(data)

                # perc. of time we do no guidance
                if np.random.random() < 0.1:
                    targets = None

                # Make the gradients zero to avoid the gradient being a
                # combination of the old gradient and the next
                self.optimizer.zero_grad()

                # model pred of noise
                eps_theta = self.eps_model(xt, t, targets)

                # calculate loss
                loss = self.loss_fn(noise, eps_theta)

            # Scale gradients
            self.scaler.scale(loss).backward()

            # Update optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Add total batch loss to total loss
            batch_loss = loss.item() * data.size(0)
            train_loss += batch_loss

            # Update info in progress bar
            process_time = start_time - time() - prepare_time
            compute_efficency = process_time / (process_time + prepare_time)
            pbar.set_description(
                f"Compute efficiency: {compute_efficency:.2f}, "
                f"batch loss: {batch_loss:.4f}, "
                f"train loss: {train_loss:.4f}"
            )
            start_time = time()

        # Calculate average loss
        train_loss /= len(self.train_loader)

        return train_loss

    def eval_step(self):

        self.eps_model.eval()

        valid_loss: float = 0.0

        pbar = tqdm(
            enumerate(self.val_loader),
            total=len(self.val_loader),
            desc="val step",
        )

        # .inference_mode() should be faster than .no_grad()
        # but you can't use .requires_grad() in that context
        with torch.inference_mode():
            for _, (data, targets) in pbar:
                data, targets = data.to(self.device), targets.to(self.device)

                # diffusion model forward process
                noise, xt, t = self.diffusion_model(data)

                # eps_model pred with cfg
                eps_theta = self.eps_model(xt, t, targets)

                if self.cfg_scale > 0:
                    # eps_model pred without cfg
                    eps_theta_uncond = self.eps_model(xt, t, None)

                    # linear interpolation between the two
                    eps_theta = torch.lerp(eps_theta_uncond, eps_theta, self.cfg_scale)

                # Calc. and acc. loss
                loss = self.loss_fn(noise, eps_theta)
                valid_loss += loss.item() * data.size(
                    0
                )  # * data.size(0) to get total loss for the batch and not the avg.

            # Calculate average loss
            valid_loss /= len(self.val_loader)

        return valid_loss

    def train(self):
        """
        ### Training loop
        """
        results = {"train_losses": list(), "valid_losses": list()}

        for epoch in range(1, self.epochs + 1):
            start = time()
            train_loss = round(self.train_step(), 4)
            valid_loss = round(self.eval_step(), 4)
            stop = time()

            print(
                f"\nEpoch: {epoch+1}",
                f"\navg train-loss: {train_loss}",
                f"\navg val-loss: {valid_loss}",
                f"\ntime: {stop-start:.4f}\n",
            )

            # Save losses
            results["train_losses"].append(train_loss),
            results["valid_losses"].append(valid_loss)

            # Log results to wandb
            self.run.log({"train_loss": train_loss, "epoch": epoch})
            self.run.log({"val_loss": valid_loss, "epoch": epoch})

            # Checkpoint saving - needs testing
            ckpt = {
                "model": self.eps_model.state_dict(),
                "epoch": epoch,
                "optim": self.optimizer.state_dict(),
            }
            save_checkpoint(
                ckpt, f"{self.run.config.checkpoints_path}/{self.run.config.name}"
            )


class AutoEncoderTrainer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        pass
