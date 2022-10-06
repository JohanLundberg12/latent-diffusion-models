import argparse
import torch


def create_parser():
    parser = argparse.ArgumentParser()

    # experiment specific args
    parser.add_argument("-ds", "--dataset", type=str, help="Training data")
    parser.add_argument(
        "-dft",
        "--diffusion_type",
        default="pixel",
        type=str,
        help="This is the type of space the forward diffusion happens at, latent or pixel space",
    )
    parser.add_argument(
        "-nt",
        "--noise_type",
        default="gaussian",
        type=str,
        help="This is the type of noise applied in the forward diffusion process",
    )

    # universal experiment settings
    parser.add_argument(
        "-t",
        "--time_steps",
        default=10,
        type=int,
        help="This is the number of steps T during the forward process",
    )
    parser.add_argument("-b", "--batch_size", default=2, type=int)
    parser.add_argument("-e", "--epochs", default=2, type=int)

    return parser


def get_device():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using {device} as backend")
    return device
