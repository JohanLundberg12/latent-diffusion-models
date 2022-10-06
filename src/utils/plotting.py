from typing import List
import matplotlib.pyplot as plt
import numpy as np


def plot(images: List, save=False, save_path=None):
    if not isinstance(images[0], list):
        images = [images]  # make 2d

    num_rows = len(images)
    num_cols = len(images[0])
    fig, axs = plt.subplots(
        figsize=(10, 10), nrows=num_rows, ncols=num_cols, squeeze=False
    )
    for row_idx, row in enumerate(images):
        for col_idx, img in enumerate(row):
            ax = axs[row_idx, col_idx]
            ax.imshow(np.asarray(img))
            ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    plt.tight_layout()

    if save:
        plt.savefig(save_path)
    else:
        plt.show()


# def viz_of_forward_process(image_path: str, image_size=64):
# image = Image.open(image_path)
# image_transform = get_image_transform(image_size=image_size)
# reverse_image_transform = get_reverse_image_transform()
# x_0 = image_transform(image, image_transform)

# model = Diffusion(n_steps=100, device=None)

# images = []

# for t in range(0, model.n_steps, 10):
# x_t = model.q_sample(x_0, torch.tensor(t))
# image_noisy = reverse_transform(x_t, reverse_image_transform)
# images.append(image_noisy)

# plot(images)
