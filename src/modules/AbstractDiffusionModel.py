from torch import nn

# class that can be either a pixel diffusion model or latent diffusion model


class AbstractDiffusionModel(nn.Module):
    def __init__(
        self,
        diffusion_type,
        noise_type,
        beta_schedule,
    ) -> None:
        super().__init__()

    def q_xt_x0():
        pass

    def q_sample():
        pass

    def p_sample():
        pass

    def forward():
        pass
