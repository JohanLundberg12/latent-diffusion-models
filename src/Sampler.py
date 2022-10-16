import os.path
import torch
import wandb

from .DDPM import Diffusion
from .UNet import UNet
from .transforms import get_reverse_image_transform, reverse_transform


class Sampler:
    def __init__(
        self,
        name: str,
        config: dict,
        diffusion_model: Diffusion,
        eps_model: UNet,
        classes=list(),
    ) -> None:
        self.name = name
        self.config = config
        self.diffusion_model = diffusion_model
        self.eps_model = eps_model
        self.classes = classes

    def sample(self, img_size):
        """
        ### Sample images
        """
        num_classes = len(self.classes)
        img_channels = self.eps_model.image_channels
        shape = (num_classes, img_channels, img_size, img_size)

        with torch.inference_mode():
            xt = self.diffusion_model.sample(
                self.eps_model,
                self.classes,
                shape=shape,
                device=self.diffusion_model.device,
            )

            transform = get_reverse_image_transform()

            for i, img_tensor in enumerate(xt):
                image = reverse_transform(img_tensor, transform)
                image.save(
                    os.path.join(
                        str(self.config.results_path), f"sampled_image_{i}.png"
                    )
                )
                image = wandb.Image(image, caption=f"sampled image_{i}")
                wandb.log({"examples": image})
