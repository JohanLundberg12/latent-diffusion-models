"""Diffusion process using some noise type
"""
from typing import Optional, Tuple

import torch
from torch import nn
from tqdm import tqdm


def gather(inp: torch.Tensor, t: torch.tensor):
    """
    Gather values in inp by applying an index across
    a dimension of the tensor.
    """
    c = inp.gather(dim=-1, index=t)

    return c.reshape(-1, 1, 1, 1)


class Diffusion(nn.Module):
    def __init__(self, n_steps: int, device: torch.device, n_samples: int = 1):
        super().__init__()
        self.device = device

        # number of images to generate when prompted to generate
        self.n_samples = n_samples

        # Create beta_1 to beta_T linearly increasing variance schedule
        self.beta = torch.linspace(0.0001, 0.02, n_steps).to(self.device)

        # alpha_t = 1 - \beta_t
        self.alpha = 1.0 - self.beta

        # \bar\alpha_t = \prod_{s=1}^t\alpha_s
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        # T
        self.n_steps = n_steps

        # \sigma^2 = beta
        self.sigma2 = self.beta

    # Get q(x_t|x_0) distribution
    def q_xt_x0(
        self, x0: torch.Tensor, t: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # gather \alpha_t and compute \sqrt(\bar\alpha_t)x_0
        mean = gather(self.alpha_bar, t) ** 0.5 * x0

        # (1 - \bar\alpha_t)I
        var = 1 - gather(self.alpha_bar, t)

        return mean, var

    # sample from q(x_t|x_0)
    def q_sample(
        self, x0: torch.Tensor, t: torch.Tensor, eps: Optional[torch.Tensor] = None
    ):
        # \epsilon ~ N(0,I)
        if eps is None:
            eps = torch.randn_like(x0)

        # get q(x_t|x_0)
        mean, var = self.q_xt_x0(x0, t)

        return mean + (var**0.5) * eps

    # sample from p_\theta(x_{-1}|x_t)
    def p_sample(self, xt: torch.Tensor, t: torch.Tensor, eps_theta):

        # gather \bar\alpha_t
        alpha_bar = gather(self.alpha_bar, t)

        # gather \alpha_t
        alpha = gather(self.alpha, t)

        # coeffiecient we scale our prediction of epsilon by
        eps_coef = (1 - alpha) / (1 - alpha_bar) ** 0.5

        # finding mu
        mean = 1 / (alpha**0.5) * (xt - eps_coef * eps_theta)

        # sigma^2
        var = gather(self.sigma2, t)

        # sample epsilon (noise)
        eps = torch.randn(xt.shape, device=xt.device)

        # finally return prediction of x_{t-1}

        return mean + (var**0.5) * eps

    @torch.no_grad()
    def sample(self, eps_model, classes, shape, device, cfg_scale=3):
        """Args:
        eps_model (Unet): Model to predict noise
        classes (tensor): tensor of all the classes
        shape (tuple): (batch, channels, image_size, image_size)
        cfg_scale (int, optional): If bigger > 0 do interpolation between
        conditional and unconditional prediction.
        """

        xt = torch.randn(shape, device=device)
        time_steps = tqdm(
            reversed(range(0, self.n_steps)),
            total=self.n_steps,
            desc="sampling loop time step",
            position=0,
        )
        for t in time_steps:
            t = (torch.ones(shape[0]) * t).long().to(device)

            # model prediction of the noise \epsilon_\theta(x_t, t, y)
            eps_theta = eps_model(xt, t, classes)  # cond pred.
            if cfg_scale > 0:
                eps_theta_uncond = eps_model(xt, t, None)  # uncond pred

                # linear interpolation of the two (start + weight * (end - start))
                eps_theta = torch.lerp(eps_theta_uncond, eps_theta, cfg_scale)

            xt = self.p_sample(xt, t, eps_theta)

        xt = xt.detach().cpu()

        return xt

    # pred noise
    def forward(self, x0: torch.Tensor, noise: Optional[torch.Tensor] = None):
        # input is shape[batch_size, ..., ..., channels]
        batch_size = x0.shape[0]

        # get random t for each sample in batch
        t = torch.randint(
            0, self.n_steps, (batch_size,), device=x0.device, dtype=torch.long
        )

        # sample noise
        if noise is None:
            noise = torch.randn_like(x0)

        # sample x_t for q(x_t|x_0)
        xt = self.q_sample(x0, t, eps=noise)

        return noise, xt, t
