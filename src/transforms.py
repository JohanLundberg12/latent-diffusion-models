import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import Lambda


def get_image_transform(image_size):
    """Returns a transform that scales the image pixel values to [-1, 1]"""

    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # Shape: HWC, Scales data into [0,1] by div / 255, so does normalization
            transforms.Lambda(lambda t: (t * 2) - 1),  # Scale between [-1, 1]
        ]
    )

    return transform


def get_reverse_image_transform():
    """Returns a transform that scales the image pixel values to [0, 255]
    and converts it to a PIL image."""

    reverse_transform = transforms.Compose(
        [
            Lambda(lambda t: (t + 1) / 2),
            Lambda(lambda t: t.permute(1, 2, 0)),  # CHW to HWC
            Lambda(lambda t: t * 255.0),  # back to 0-255
            Lambda(lambda t: t.cpu().numpy().astype(np.uint8)),
            transforms.ToPILImage(),
        ]
    )
    return reverse_transform


def image_transform(image: Image.Image, transform) -> torch.Tensor:
    """Transforms an image to a tensor and scales its pixel values to be between -1 and 1."""

    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)  # Reshape to (B, C, H, W)

    return image_tensor


def reverse_transform(image_tensor, transform) -> Image.Image:
    """Transforms a tensor to an image and scales its pixel values to be between 0 and 255."""

    if len(image_tensor.shape) == 4:
        image_tensor = image_tensor.squeeze()  # Reshape to (C, H, W)

    image = transform(image_tensor)

    return image
