"""generates images from a trained model"""
# This script:
# 1 - Loads the UNet model
# 2 - Loads the diffusion model
# 3 - Generates images
# 4 - Saves the images


import sys

import torch
import yaml

from src.utils import create_folder, get_model_from_config
from src.utils import get_device, save_images


def main(model, diffusion_model, config, device):
    # Set the shape of the images to be generated
    shape = (
        1,
        config["data"]["image_channels"],
        config["data"]["image_size"],
        config["data"]["image_size"],
    )

    folder = f"{config['diffusion']['type']}/{config['project_name']}/results"

    for i in range(model.num_classes):
        class_name = str(i)
        create_folder(f"{folder}/{class_name}")

        tensor_image = diffusion_model.sample(
            model,
            torch.tensor(list(range(i, i + 1))).to(device),
            shape=shape,
            device=device,
            cfg_scale=3,
        )

        save_images(tensor_image, f"{folder}/{class_name}/sample")


# checkpoint is in the folder pixel/name/checkpoints
def get_state_dict_path_from_config(config):
    return f"{config['diffusion']['type']}/{config['project_name']}/checkpoints/checkpoint.pt"


if __name__ == "__main__":
    config = yaml.safe_load(open(sys.argv[1], "r"))
    state_dict_path = get_state_dict_path_from_config(config)

    device = get_device()

    diffusion_model = get_model_from_config(config["diffusion"])
    model = get_model_from_config(config["model"], state_dict_path=state_dict_path).to(
        device
    )

    main(model, diffusion_model, config, device)
