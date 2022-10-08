import numpy as np
import sys
import yaml
import torch

from dataclasses import dataclass

from modules.DDPM import Diffusion
from modules.Unet import UNet
from src.utils.utils import update_config
from utils.transforms import get_reverse_image_transform, reverse_transform
from modules.utils import instantiate_model


@dataclass
class Sampler:
    name: str
    config: dict
    diffusion_model: Diffusion
    eps_model: UNet

    def sample(self):
        """
        ### Sample images
        """
        classes = self.eps_model.classes
        num_classes = len(classes)
        img_channels = self.eps_model.image_channels
        img_size = self.eps_model.image_size
        shape = (num_classes, img_channels, img_size, img_size)

        with torch.no_grad():
            xt = self.diffusion_model.sample(
                self.eps_model, classes, shape=shape, device=self.diffusion_model.device
            )

            transform = get_reverse_image_transform()

            for i, img_tensor in enumerate(xt):
                image = reverse_transform(img_tensor, transform)
                image.save(
                    str(self.config.results_path) + f"sampled_image_{i}.png",
                )


if __name__ == "__main__":
    config = sys.argv[1]

    # Get config experiment configurations
    config_file = sys.argv[1]
    config = yaml.safe_load(open(config_file, "r"))
    config = update_config(config)

    # set seeds
    config.seed = 42
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    diffusion_model = Diffusion(
        n_steps=config["n_steps"],
        device=config.device,
    )
    eps_model = instantiate_model(config, load=True)

    sampler = Sampler(
        name=config["name"],
        config=config,
        diffusion_model=diffusion_model,
        eps_model=eps_model,
    )
