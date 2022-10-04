from typing import List
import errno
import pathlib
from pathlib import Path
import inspect
import os.path
import yaml
from .utils import get_device


def _create_folder(path: pathlib.PosixPath) -> None:
    """Args:
    path (pathlib.Path): relative path to be created"""

    try:
        Path.mkdir(path, parents=True, exist_ok=True)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def _get_caller_file() -> str:
    frames: List[inspect.FrameInfo] = inspect.stack()
    project_root = Path(__file__).absolute().parent

    for f in frames:
        module_path = Path(f.filename)
        if str(module_path).startswith(os.path.join(project_root, "")):
            continue
        if str(module_path).startswith("<stdin"):
            break
        if str(module_path).startswith("<ipython"):
            break
        if not module_path.exists():
            break
        return str(module_path)

    return str(Path("").absolute())


class Experiment:
    """
    Experiment contains path specific properties
    and config settings.
    """

    def __init__(self):
        self.path = Path(_get_caller_file()).parent.resolve()
        self.data_path: Path = self.path / "data"
        self.models_path: Path = self.path / "models"
        self.results_path: Path = self.path / "results"
        self.config_file_path: Path = None
        self.config: dict = None

        self.device: str = get_device()
        self.experiment_name: str = None

        _create_folder(self.data_path)
        _create_folder(self.models_path)
        _create_folder(self.results_path)

    def load_config_file(self, config_file_path: Path):
        self.config_file_path = config_file_path
        config = yaml.safe_load(open(config_file_path, "r"))
        self.config = config
        self.experiment_name = self.config["name"]

        return config

    def get_config(self):
        return self.config

    def __str__(self):
        return f"<Experiment path={self.path}>"

    def __repr__(self):
        return str(self)

    def get_config_file_path(self):
        return self.config_file_path

    def get_data_path(self):
        return self.data_path

    def get_models_path(self):
        return self.models_path

    def get_results_path(self):
        return self.results_path

    def get_device(self):
        return self.device
