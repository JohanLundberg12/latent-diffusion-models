import sys
import torch
from dataclasses import dataclass
from utils.data_utils import Experiment
from models.Diffusion import Diffusion
from models.Unet import UNet
from utils.transforms import get_reverse_image_transform, reverse_transform
from models.utils import instantiate_model


@dataclass
class Sampler:
    name: str
    experiment: Experiment
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
                    str(self.experiment.results_path) + f"sampled_image_{i}.png",
                )


if __name__ == "__main__":
    config = sys.argv[1]

    # Create experiment
    global _experiment_singleton
    _experiment_singleton = Experiment()

    # Get config experiment configurations
    config_file = sys.argv[1]
    config = _experiment_singleton.load_config_file(config_file)

    diffusion_model = Diffusion(
        n_steps=config["n_steps"],
        device=_experiment_singleton.device,
    )
    eps_model = instantiate_model(_experiment_singleton, load=True)

    sampler = Sampler(
        name=config["name"],
        experiment=_experiment_singleton,
        diffusion_model=diffusion_model,
        eps_model=eps_model,
    )
