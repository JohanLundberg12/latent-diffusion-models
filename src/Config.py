from .utils import create_folder, get_device, set_seed


class Config:
    """Class to store the configurations."""

    def __init__(self, **entries):
        self.__dict__.update(entries)

        self.__dict__.update({"device": get_device(), "seed": set_seed()})

        self.__dict__.update(
            {"dirpath": f"{self.__dict__['type']}/{self.__dict__['project_name']}"}
        )
        create_folder(self.__dict__["dirpath"])

        self.__dict__.update({"results": f"{self.__dict__['dirpath']}/results"})
        create_folder(self.__dict__["results"])

        self.__dict__.update({"checkpoints": f"{self.__dict__['dirpath']}/checkpoints"})
        create_folder(self.__dict__["checkpoints"])
