from typing import Union
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from src.DDPM import Diffusion
from src.transforms import (
    get_image_transform,
    get_reverse_image_transform,
    image_transform,
    reverse_transform,
)


def test_forward_diffusion(
    img: Union[str, torch.Tensor],
    t: Union[torch.Tensor, int],
    diffusion_model: Diffusion = None,
    device="cpu",
    image_size: int = 128,
):
    """Test the forward diffusion process of a DDPM model.
    The image can be a file or a tensor. If file it will be loaded and transformed.
    The diffusion model can either be given or it has to be instantiated.
    t is the time step that will be sampled but it will also be used to create the model, thus
    to avoid out of bounds error, t at sample time must be smaller than t at instantiation time.
    """
    if isinstance(img, str):
        img = Image.open(img)
        transform = get_image_transform(image_size)
        img = image_transform(img, transform)

    if not isinstance(diffusion_model, Diffusion):
        # +1 to avoid index 5 is out of bounds for dimension 0 with size 5
        diffusion_model = Diffusion(int(t) + 1, device=device)

    if isinstance(t, int):
        t = torch.tensor([t])
    img_noisy = diffusion_model.q_sample(img, t)  # Forward process
    transform = get_reverse_image_transform()
    img = reverse_transform(img_noisy, transform=transform)

    return img


def plot_forward_process(images, save_path=None):
    if not isinstance(images[0], list):
        images = [images]  # make 2d

    num_rows = len(images)  # number of images
    num_cols = len(images[0])  # number of time steps

    fig, axs = plt.subplots(
        figsize=(10, 10), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def viz_of_forward_process(image_path: str, image_size=64, n_steps=100):
    image = Image.open(image_path)
    img_transform = get_image_transform(image_size=image_size)
    x_0 = image_transform(image, img_transform)

    model = Diffusion(n_steps=n_steps, device=None)

    images = []

    # test forward process with stepsize of 10
    for t in range(0, model.n_steps, 10):
        x_t = test_forward_diffusion(x_0, torch.tensor(t), diffusion_model=model)
        images.append(x_t)

    plot_forward_process(images)
