from typing import Callable

import torch
import torch.nn.functional as f
import torch.utils.data
import torch.optim as optim
import wandb

from .EarlyStopping import EarlyStopping


# Reconstruction + KL divergence losses summed over all elements and batch
def elbo_loss_function(recon_x, x, mu, logvar):
    bce = f.binary_cross_entropy(recon_x, x, reduction="sum")
    # why binary cross entropy? because we are using a bernoulli distribution
    # see https://stats.stackexchange.com/questions/338904/why-is-the-loss-function-for-variational-autoencoders-binary-cross-entropy

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return bce + kld


# abstract class for DiffusionModelTrainer, LatentDiffusionModelTrainer and ResNetTrainer
# parameters: config, model, train_loader, val_loader
# optimizer, scaler, device and epochs created in init or inferred from config
class Trainer:
    def __init__(self, config, model, train_loader, val_loader, classes):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.classes = classes

        self.optimizer = self._get_optimizer()

        # instances of GradScaler() help perform steps of the gradient scaling
        # conveniently. Improves convergence for networks with
        # float16 (Conv layers) gradients by minimizing gradient underflow
        self.scaler = torch.cuda.amp.GradScaler() if config["use_amp"] else None

        self.device = config["device"]
        self.epochs = config["epochs"]
        self.loss_fn = self._get_loss_fn()
        self.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{self.config.checkpoints}/checkpoint.pt",
        )

    # get loss function
    def _get_loss_fn(self) -> Callable:
        """Returns a loss function"""
        loss_fn = self.config["loss_fn"]

        if loss_fn == "mse":
            return f.mse_loss
        elif loss_fn == "elbo":
            return elbo_loss_function
        elif loss_fn == "cross-entropy":
            return f.cross_entropy
        else:
            raise NotImplementedError

    def _get_optimizer(self) -> optim.Optimizer:
        optimizer = optim.Adam(self.model.parameters(), lr=self.config["lr"])

        return optimizer

    def _train_epoch(self, epoch: int):
        raise NotImplementedError

    def _val_epoch(self, epoch: int):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def _log_metrics(self, metrics: dict, step: int):
        """Logs metrics to wandb"""
        for key, value in metrics.items():
            wandb.log({f"{key}": value}, step=step)

    # wandb logging of images using a list comprehension
    def _log_images(self, images: list, step: int, mode: str):
        """Logs images to wandb"""
        wandb.log(
            {f"{mode}/images": [wandb.Image(image) for image in images]}, step=step
        )
