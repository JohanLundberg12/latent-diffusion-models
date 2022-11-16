from typing import Callable, Union
from time import time
import numpy as np

import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.data import DataLoader

import wandb

from .DDPM import Diffusion
from .UNet import UNet
from .LatentDiffusionModel import LatentDiffusionModel
from .EarlyStopping import EarlyStopping

from src.transforms import (
    get_reverse_image_transform,
    reverse_transform,
)
from .utils import progress_bar


# ddpm and autoencoder
class DiffusionModelTrainer:
    def __init__(
        self,
        config: dict,
        diffusion_model: Diffusion,
        eps_model: Union[UNet, LatentDiffusionModel],
        train_loader: DataLoader,
        val_loader: DataLoader,
        cfg_scale: int,  # classifier free guidance scale
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler,
        classes: list(),
        device: str,
        epochs: int,
    ) -> None:
        self.config = config
        self.device = device
        self.epochs = epochs
        self.classes = classes

        self.diffusion_model = diffusion_model.to(self.device)
        self.eps_model = eps_model.to(self.device)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.cfg_scale = cfg_scale  # classifier free guidance
        self.classes = torch.tensor(self.classes).to(self.device)

        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler
        self.early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path=self.config.data_paths["checkpoints"] + "/checkpoint.pt",
        )

    def train_step(self):
        """
        ### Train
        """
        self.eps_model.train()

        train_loss: float = 0.0

        pbar = progress_bar(self.train_loader, desc="train step")

        start_time = time()

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            with torch.cuda.amp.autocast():

                # if latent noise predictor model
                if isinstance(self.eps_model, LatentDiffusionModel):

                    # encode image
                    z = self.eps_model.autoencoder_encode(data)

                    # latent diffusion model forward pass: add noise to encoding
                    noise, xt, t = self.diffusion_model(z)
                else:
                    # pixel diffusion model forward pass
                    noise, xt, t = self.diffusion_model(data)

                # perc. of time we do no guidance
                if np.random.random() < 0.1:
                    targets = None

                # model pred of noise
                eps_theta = self.eps_model(xt, t, targets)

                # calculate loss
                loss = self.loss_fn(noise, eps_theta)

            # Make the gradients zero to avoid the gradient being a
            # combination of the old gradient and the next
            # Updates gradients by write rather than read and write (+=) used
            # https://www.youtube.com/watch?v=9mS1fIYj1So
            self.optimizer.zero_grad(set_to_none=True)

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

        pbar = progress_bar(self.val_loader, desc="val step")

        # .inference_mode() should be faster than .no_grad()
        # but you can't use .requires_grad() in that context
        with torch.inference_mode():
            for _, (data, targets) in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                if isinstance(self.eps_model, LatentDiffusionModel):
                    # encode image
                    z = self.eps_model.autoencoder_encode(data)

                    # latent diffusion model forward pass: add noise to encoding
                    noise, xt, t = self.diffusion_model(z)
                else:
                    # diffusion model forward pass
                    noise, xt, t = self.diffusion_model(data)

                # model pred of noise with cfg
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
                f"\nEpoch: {epoch}",
                f"\navg train-loss: {train_loss}",
                f"\navg val-loss: {valid_loss}",
                f"\ntime: {stop-start:.4f}\n",
            )

            # Save losses
            results["train_losses"].append(train_loss),
            results["valid_losses"].append(valid_loss)

            # Log results to wandb
            wandb.log({"train_loss": train_loss, "epoch": epoch}, step=epoch)
            wandb.log({"val_loss": valid_loss, "epoch": epoch}, step=epoch)

            if epoch % 2 == 0:
                tensor_image = self.diffusion_model.sample(
                    self.eps_model,
                    self.classes,
                    shape=(
                        len(self.classes),
                        self.config.data["image_channels"],
                        self.config.data["image_size"],
                        self.config.data["image_size"],
                    ),
                    device=self.device,
                    cfg_scale=self.cfg_scale,
                )
                transform = get_reverse_image_transform()
                images = [
                    reverse_transform(image, transform=transform)
                    for image in tensor_image
                ]

                wandb.log(
                    {"images": [wandb.Image(image) for image in images]},
                    step=epoch,
                )

            self.early_stopping(val_loss=valid_loss, model=self.eps_model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
