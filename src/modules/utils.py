import os
import shutil
import torch
from pathlib import Path

from .Unet import UNet


def get_model(config: dict, load=False):
    model = UNet(
        image_channels=config["image_channels"],
        n_channels=config["n_channels"],
        ch_mults=config["channel_multipliers"],
        out_channels=config["out_channels"],
        with_time_emb=config["with_time_emb"],
        num_classes=config["num_classes"],
    )
    if load:
        if isinstance(config.models_path, str):
            config.models_path = Path(config.models_path)
        model = model.load_state_dict(torch.load(config.models_path / config.name))

    return model.to(config.device)


def save_checkpoint(state, save_path: str, is_best: bool = False, max_keep: int = 10):
    """Saves torch model to checkpoint file.
    Args:
        state (torch model state): State of a torch Neural Network
        save_path (str): Destination path for saving checkpoint
        is_best (bool): If ``True`` creates additional copy
            ``best_model.ckpt``
        max_keep (int): Specifies the max amount of checkpoints to keep
    """

    # deal with max_keep
    save_dir = os.path.dirname(save_path)
    list_path = os.path.join(save_dir, "latest_checkpoint.txt")

    save_path_base_name = os.path.basename(save_path)

    if os.path.exists(list_path):
        with open(list_path) as f:
            ckpt_list = f.readlines()
            len_list = str(len(ckpt_list))
            save_path_base_name = f"{save_path_base_name}_{len_list}.ckpt"
            ckpt_list = [f"{save_path_base_name}" + "\n"] + ckpt_list
    else:
        len_list = "0"
        save_path_base_name = save_path_base_name + "_" + len_list + ".ckpt"
        ckpt_list = [f"{save_path_base_name}" + "\n"]

    # save checkpoint
    torch.save(state, save_dir + "/" + save_path_base_name)

    if max_keep is not None:
        for ckpt in ckpt_list[max_keep:]:
            ckpt = os.path.join(save_dir, ckpt[:-1])
            if os.path.exists(ckpt):
                os.remove(ckpt)
        ckpt_list[max_keep:] = []

    with open(list_path, "w") as f:
        f.writelines(ckpt_list)

    # copy best
    if is_best:
        shutil.copyfile(save_path, os.path.join(save_dir, "best_model.ckpt"))


def load_checkpoint(ckpt_dir_or_file: str, map_location=None, load_best=False):
    """Loads torch model from checkpoint file.
    Args:
        ckpt_dir_or_file (str): Path to checkpoint directory or filename
        map_location: Can be used to directly load to specific device
        load_best (bool): If True loads ``best_model.ckpt`` if exists.
    """
    if os.path.isdir(ckpt_dir_or_file):
        if load_best:
            ckpt_path = os.path.join(ckpt_dir_or_file, "best_model.ckpt")
        else:
            with open(os.path.join(ckpt_dir_or_file, "latest_checkpoint.txt")) as f:
                ckpt_path = os.path.join(ckpt_dir_or_file, f.readline()[:-1])
    else:
        ckpt_path = ckpt_dir_or_file
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print(" [*] Loading checkpoint from %s succeed!" % ckpt_path)
    return ckpt
