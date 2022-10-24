from typing import Callable
from time import time
import wandb

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from src.transforms import get_reverse_image_transform, reverse_transform

from src.utils import progress_bar

from .EarlyStopping import EarlyStopping
from .Autoencoder import Autoencoder

from src import save_images


class AutoencoderTrainer:
    def __init__(
        self,
        config: dict,
        model: Autoencoder,
        train_loader: DataLoader,
        val_loader: DataLoader,
        loss_fn: Callable,
        optimizer: optim.Optimizer,
        scaler: torch.cuda.amp.grad_scaler,
        classes: list(),
        device: str,
        epochs: int,
    ) -> None:

        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scaler = scaler
        self.classes = classes
        self.epochs = epochs
        self.device = device
        self.early_stopping = EarlyStopping(
            patience=10,
            verbose=True,
            path=self.config.data_paths["checkpoints"] + "/checkpoint.pt",
        )
        self.reconstruction_loss_factor = 1

    def train_step(self, epoch):
        self.model.train()

        train_loss = 0.0

        start_time = time()

        pbar = progress_bar(self.train_loader, desc="train step")

        for _, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)
            prepare_time = start_time - time()

            # Autocasting automatically chooses the precision (floating point data type)
            # for GPU operations to improve performance while maintaining accuracy.
            with torch.cuda.amp.autocast():
                outputs, mu, sigma = self.model(data)

                loss = self.loss_fn(outputs, data, mu, sigma)

            self.optimizer.zero_grad()

            # The network loss is scaled by a scaling factor to prevent underflow.
            # Gradients flowing back through the network are scaled by the same factor.
            # Calls .backward() on scaled loss to create scaled gradients.
            self.scaler.scale(loss).backward()

            # Scaler.step() first unscales the gradients of the optimizer's
            # assigned params by dividing them by the scale factor.
            # If the gradients do not contain NaNs/inf, optimizer.step() is called,
            # otherwise skipped.
            # optimizer.step() is then called using the unscaled gradients.
            self.scaler.step(self.optimizer)

            # Updates the scale factor
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

    def eval_step(self, epoch):
        self.model.eval()

        valid_loss: float = 0.0

        pbar = progress_bar(self.val_loader, desc="val step")

        # .inference_mode() should be faster than .no_grad()
        # but you can't use .requires_grad() in that context
        with torch.inference_mode():
            for _, (data, targets) in pbar:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                # Calc. and acc. loss
                loss = self.loss_fn(
                    outputs,
                    data,
                    self.model.distribution.mean,
                    self.model.distribution.log_var,
                )
                valid_loss += loss.item() * data.size(
                    0
                )  # * data.size(0) to get total loss for the batch and not the avg.

            # Calculate average loss
            valid_loss /= len(self.val_loader)

        if epoch % 5 == 0:
            # save the reconstructed images
            save_images(outputs, name=self.config.data_paths["results"] + "/image")

            transform = get_reverse_image_transform()
            images = [
                reverse_transform(image, transform=transform) for image in outputs
            ]

            for i, img in enumerate(images):
                wandb.log(
                    {
                        f"Sample image {i} at epoch: {epoch}": wandb.Image(
                            img, caption=f"image {i}"
                        )
                    }
                )

        return valid_loss

    def train(self):
        """
        ### Training loop
        """
        results = {"train_losses": list(), "valid_losses": list()}

        for epoch in range(1, self.epochs + 1):
            start = time()
            train_loss = round(self.train_step(epoch), 4)
            valid_loss = round(self.eval_step(epoch), 4)
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
            wandb.log({"train_loss": train_loss, "epoch": epoch})
            wandb.log({"val_loss": valid_loss, "epoch": epoch})

            self.early_stopping(val_loss=valid_loss, model=self.model)

            if self.early_stopping.early_stop:
                print("Early stopping")
                break
