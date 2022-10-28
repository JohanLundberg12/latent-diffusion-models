import torch
from torch import nn

from src.UNet import UNet


class DiffusionWrapper(nn.Module):
    """
    *This is an empty wrapper class around the [U-Net](model/unet.html).
    We keep this to have the same model structure as
    [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
    so that we do not have to map the checkpoint weights explicitly*.
    """

    def __init__(self, diffusion_model: UNet):
        super().__init__()
        self.diffusion_model = diffusion_model

    def forward(self, x: torch.Tensor, time_steps: torch.Tensor, targets: torch.Tensor):
        return self.diffusion_model(x, time_steps, targets)


class LatentDiffusionModel(nn.Module):
    def __init__(
        self,
        eps_model,
        autoencoder,
        latent_scaling_factor: float,
        n_steps: int,
        linear_start: float,
        linear_end: float,
    ):
        super().__init__()

        self.model = DiffusionWrapper(eps_model)
        self.autoencoder = autoencoder
        self.latent_scaling_factor = latent_scaling_factor
        self.n_steps = n_steps

        # $\beta$ schedule
        beta = (
            torch.linspace(
                linear_start**0.5, linear_end**0.5, n_steps, dtype=torch.float64
            )
            ** 2
        )
        self.beta = nn.Parameter(beta.to(torch.float32), requires_grad=False)

        # $\alpha_t = 1 - \beta_t$
        alpha = 1.0 - beta

        # $\bar\alpha_t = \prod_{s=1}^t \alpha_s$
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.alpha_bar = nn.Parameter(alpha_bar.to(torch.float32), requires_grad=False)

    def autoencoder_encode(self, image: torch.Tensor):
        """
        ### Get scaled latent space representation of the image
        The encoder output is a distribution.
        We sample from that and multiply by the scaling factor.
        """

        # their Autoencoder encode method returns a class of type GaussianDistribution which has a .sample method
        return self.latent_scaling_factor * self.autoencoder.encode(image).sample()

    def autoencoder_decode(self, z: torch.Tensor):
        """
        ### Get image from the latent representation
        We scale down by the scaling factor and then decode.
        """
        return self.first_stage_model.decode(z / self.latent_scaling_factor)

    def forward(self, x: torch.Tensor, t: torch.Tensor, targets: torch.Tensor):
        """
        ### Predict noise
        Predict noise given the latent representation $x_t$, time step $t$, and the
        conditioning context $c$.
        $$\epsilon_\text{cond}(x_t, c)$$
        """
        return self.model(x, t, targets)
