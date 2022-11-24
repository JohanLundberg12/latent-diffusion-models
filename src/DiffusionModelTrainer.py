import torch
import numpy as np

from .utils import progress_bar, timeit

from .Trainer import Trainer

from .transforms import (
    get_reverse_image_transform,
    reverse_transform,
)


class DiffusionModelTrainer(Trainer):
    def __init__(
        self, config, model, train_loader, val_loader, classes, diffusion, cfg_scale
    ):
        super().__init__(config, model, train_loader, val_loader, classes)

        self.diffusion = diffusion
        self.cfg_scale = cfg_scale

        # put model and diffusion on device
        self.to(self.device)
        self.classes = torch.tensor(self.classes).to(self.device)

    @timeit
    def _train_epoch(self, epoch):
        self.model.train()

        train_loss = 0.0
        pbar = progress_bar(
            self.train_loader, desc=f"Train, Epoch {epoch + 1}/{self.epochs}"
        )

        for i, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                noise, xt, t = self.diffusion(data)

                # % of the time we don't use labels (no guidance)
                if np.random.random() < 0.1:
                    targets = None
                eps_theta = self.forward(xt, t, targets)

                loss = self.loss_fn(noise, eps_theta)

            # zero the parameter gradients of the optimizer
            # Make the gradients zero to avoid the gradient being a
            # combination of the old gradient and the next
            # Updates gradients by write rather than read and write (+=) used
            # https://www.youtube.com/watch?v=9mS1fIYj1So
            self.optimizer.zero_grad(set_to_none=True)

            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()

            # update train loss and multiply by
            # data.size(0) to get the sum of the batch loss
            train_loss += loss.item() * data.size(0)

            # update info in progress bar
            pbar.set_description(
                f"Train, Epoch {epoch + 1}/{self.epochs}, train loss: {train_loss:.4f}"
            )

        # calculate average losses
        train_loss = train_loss / len(self.train_loader)

        return train_loss

    @timeit
    @torch.inference_mode()
    def _val_epoch(self, epoch):
        self.model.eval()

        val_loss = 0.0
        pbar = progress_bar(
            self.val_loader, desc=f"Val, Epoch {epoch + 1}/{self.epochs}"
        )

        for i, (data, targets) in pbar:
            data, targets = data.to(self.device), targets.to(self.device)

            # forward + backward + optimize
            with torch.cuda.amp.autocast(enabled=self.config["use_amp"]):
                noise, xt, t = self.diffusion(data)

                # model pred of noise with cfg
                eps_theta = self.forward(xt, t, targets)

                if self.cfg_scale > 0:
                    # eps_model pred without cfg
                    eps_theta_uncond = self.forward(xt, t, None)

                    # linear interpolation between the two
                    eps_theta = torch.lerp(eps_theta_uncond, eps_theta, self.cfg_scale)

                loss = self.loss_fn(noise, eps_theta)

            # update val loss
            val_loss += loss.item() * data.size(0)

            pbar.set_description(
                f"Val, Epoch {epoch + 1}/{self.epochs}, val loss: {val_loss:.4f}"
            )

        # calculate average losses
        val_loss = val_loss / len(self.val_loader)

        return val_loss

    @timeit
    def train(self):

        for epoch in range(self.epochs):
            train_loss = round(self._train_epoch(epoch), 4)
            val_loss = round(self._val_epoch(epoch), 4)

            print(
                f"\nEpoch {epoch + 1}/{self.epochs}, \ntrain loss: {train_loss}, \nval loss: {val_loss}\n"
            )

            # Log results to wandb
            self._log_metrics(
                metrics={"diffusion_model train_loss": train_loss},
                step=epoch,
            )
            self._log_metrics(
                metrics={"diffusion_model val_loss": val_loss}, step=epoch
            )

            if epoch % 2 == 0:
                images = self.sample(self.classes, cfg_scale=self.cfg_scale)
                self._log_images(images, step=epoch, mode="sample")
                print("Sampled images logged to wandb\n")

            # early stopping
            self.early_stopping(val_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

    def forward(self, x, t, targets=None):
        if targets is None:
            # if no targets, we don't use cfg
            eps_theta = self.model(x, t)
        else:
            # if targets, we use cfg
            eps_theta = self.model(x, t, targets)

        return eps_theta

    @timeit
    def sample(self, classes, cfg_scale=0):
        tensor_image = self.diffusion.sample(
            self.model,
            classes,
            shape=(
                len(classes),
                self.config.data["image_channels"],
                self.config.data["image_size"],
                self.config.data["image_size"],
            ),
            device=self.device,
            cfg_scale=cfg_scale,
        )
        transform = get_reverse_image_transform()
        images = [
            reverse_transform(image, transform=transform) for image in tensor_image
        ]

        return images

    def to(self, device):
        self.device = device
        self.model.to(device)
        self.diffusion.to(device)
        return self
