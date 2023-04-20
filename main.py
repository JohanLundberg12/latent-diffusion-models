# this script loads the training data, splits it into two parts 50%-50%,
# splits the first part into 90%-10% and trains the diffusion model on the 90% part
# and evaluates it on the 10% part.
# The second part is split into 90%-10% and the ResNet model is trained on the 90% part
# and evaluated on the 10% part.

import math
import numpy as np
import sys
import yaml
import wandb
import torch
import torch.nn.functional as f

import torchvision.datasets as datasets

from torch.utils.data import DataLoader
from torch.utils.data import random_split

from src.Config import Config
from src.DDPM import Diffusion
from src.UNet import UNet
from src.ResNetClassifier import ResNetBase
from src.ResNetTrainer import ResNetTrainer
from src.DiffusionModelTrainer import DiffusionModelTrainer
from src.utils import get_device, create_folder, save_images, load_model
from src.transforms import get_image_transform, get_gray_scale_image_transform
from src.EarlyStopping import EarlyStopping

import os

os.environ["WANDB_MODE"] = "dryrun"


def num_to_groups(num, group_size):
    n = num // group_size

    return n


def load_synthetic_data(path, image_size):
    transform = get_gray_scale_image_transform(image_size)

    # dataset containing (image, label)
    dataset = datasets.ImageFolder(path, transform=transform)

    return dataset


def create_model(name, dataset, num_classes):
    if name == "UNet":
        if dataset == "CIFAR10":
            model = UNet(
                in_channels=3,
                out_channels=3,
                channels=64,
                channel_multipliers=[1, 2, 4, 8],
                with_time_emb=True,
                num_classes=num_classes,
            )
        elif dataset == "MNIST":
            model = UNet(
                in_channels=1,
                out_channels=1,
                channels=64,
                channel_multipliers=[1, 2, 4, 8],
                with_time_emb=True,
                num_classes=num_classes,
            )

        return model.to(get_device())

    elif name == "ResNet":
        if dataset == "CIFAR10":
            resnet_model = ResNetBase(
                img_channels=3,
                out_channels=10,
                n_blocks=[2, 2, 2, 2],
                n_channels=[64, 128, 256, 512],
            )
        elif dataset == "MNIST":
            resnet_model = ResNetBase(
                img_channels=1,
                out_channels=10,
                n_blocks=[2, 2, 2, 2],
                n_channels=[64, 128, 256, 512],
            )

        return resnet_model.to(get_device())


def main(config: dict):
    with wandb.init(project=config.project_name, entity=config.entity, config=config):
        config = wandb.config

        transforms = get_image_transform(config.data["image_size"])
        batch_size = config["batch_size"]

        # load the training data
        if config.data["dataset"] == "CIFAR10":
            trainset = datasets.CIFAR10(
                root="./data",
                train=True,
                download=True,
                transform=transforms,
            )
            classes = list(np.unique(trainset.targets))
        elif config.data["dataset"] == "MNIST":
            trainset = datasets.MNIST(
                root="./data",
                train=True,
                download=True,
                transform=transforms,
            )
            classes = list(np.unique(trainset.targets.numpy()))
        num_classes = len(classes)

        # if debugging is enabled, only use a small subset of the data
        if config.debugging:
            trainset = torch.utils.data.Subset(trainset, range(100))

        # split the training data into two parts 50%-50%
        dataset1, dataset2 = random_split(
            trainset, [int(len(trainset) / 2), int(len(trainset) / 2)]
        )

        # split the first part, trainset1, into 90%-10%
        trainset1, valset1 = random_split(
            dataset1, [int(len(dataset1) * 0.9), int(len(dataset1) * 0.1)]
        )

        # split the second part, trainset2, into 90%-10%
        trainset2, valset2 = random_split(
            dataset2, [int(len(dataset2) * 0.9), int(len(dataset2) * 0.1)]
        )

        ##############################################################################
        # train the diffusion model
        ##############################################################################

        # load UNet model
        model = create_model("UNet", config.data["dataset"], num_classes)

        # load diffusion model from src.DDPM.py and train it on the 90% part of the first part
        # evaluate it on the 10% part of the first part
        diffusion_model = Diffusion(
            n_steps=400,
            device=get_device(),
            n_samples=int(len(trainset1) / num_classes),
        )  # we sample len(trainset1) / num_classes images per class

        # data loader
        train_loader = DataLoader(
            trainset1,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valset1,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # create trainer class and train and evaluate the model
        trainer = DiffusionModelTrainer(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=classes,
            diffusion=diffusion_model,
            cfg_scale=config.diffusion["cfg_scale"],
        )
        trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/diffusion_model.pt",
        )

        wandb.watch(model, trainer.loss_fn, log="all", log_freq=10)
        wandb.define_metric("diffusion_model train_loss", summary="min")
        wandb.define_metric("diffusion_model valid_loss", summary="min")

        trainer.train()

        ##############################################################################
        # exp1: train the ResNet model on real data
        ##############################################################################

        # load ResNet model
        resnet_model = create_model("ResNet", config.data["dataset"], num_classes)

        # train the ResNet model on the 90% part of the second part
        # evaluate it on the 10% part of the second part
        # data loader
        train_loader = DataLoader(
            trainset2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        val_loader = DataLoader(
            valset2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=classes,
        )
        resnet_trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/resnet_model exp1.pt",
        )
        # overwrite default loss function
        resnet_trainer.loss_fn = f.cross_entropy

        wandb.watch(resnet_model, resnet_trainer.loss_fn, log="all", log_freq=10)
        name = "resnet_model exp1"
        wandb.define_metric(f"{name} train_loss", summary="min")
        wandb.define_metric(f"{name} valid_loss", summary="min")
        wandb.define_metric(f"{name} train_f1", summary="max")
        wandb.define_metric(f"{name} valid_f1", summary="max")

        resnet_trainer.train(exp_name=name)

        ##############################################################################
        # evaluate the ResNet model on the test set
        ##############################################################################
        # load the test data
        if config.data["dataset"] == "CIFAR10":
            testset = datasets.CIFAR10(
                root="./data",
                train=False,
                download=True,
                transform=transforms,
            )
        elif config.data["dataset"] == "MNIST":
            testset = datasets.MNIST(
                root="./data",
                train=False,
                download=True,
                transform=transforms,
            )

        # if debugging is enabled, only use a small subset of the data
        if config.debugging:
            testset = torch.utils.data.Subset(testset, range(100))

        testloader = DataLoader(
            testset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )

        # load the model
        resnet_model = load_model(
            resnet_model,
            f"{config['checkpoints']}/resnet_model exp1.pt",
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=train_loader,
            val_loader=val_loader,
            classes=classes,
        )

        # evaluate the model on the test set

        f1_scores, avg_f1 = resnet_trainer.run(
            mode="test", dataloader=testloader, step="test step"
        )
        wandb.log({f"{name} avg_test_f1": avg_f1})
        wandb.log({f"{name} test_scores": f1_scores})

        ##############################################################################
        # generate images using the diffusion model
        ##############################################################################
        n = num_to_groups(diffusion_model.n_samples, config["batch_size"])

        shape = (
            config["batch_size"],
            config["data"]["image_channels"],
            config["data"]["image_size"],
            config["data"]["image_size"],
        )

        folder = f"{config['results']}"

        # load diffusion model checkpoint
        model = load_model(
            model,
            f"{config['checkpoints']}/diffusion_model.pt",
        )

        for i in range(model.num_classes):
            class_name = str(i)
            create_folder(f"{folder}/{class_name}")

            for j in range(n):
                tensor_image = diffusion_model.sample(
                    model,
                    torch.tensor(list(range(i, i + 1))).to(get_device()),
                    shape=shape,
                    device=get_device(),
                    cfg_scale=3,
                )

                save_images(tensor_image, f"{folder}/{class_name}/sample_{j}")

        ##############################################################################
        # exp2: train resnet model on the generated images
        ##############################################################################

        # instantiate model again
        resnet_model = create_model("ResNet", config.data["dataset"], num_classes)

        # load the generated images
        generated_images = load_synthetic_data(path=f"{folder}", image_size=32)
        generated_images_loader = DataLoader(
            generated_images,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # train the ResNet model on the generated images
        # evaluate it on the 10% part of the second part
        # val dataloader
        val_loader = DataLoader(
            valset2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=generated_images_loader,
            val_loader=val_loader,
            classes=classes,
        )
        resnet_trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/resnet_model exp2.pt",
        )
        # overwrite default loss function
        resnet_trainer.loss_fn = f.cross_entropy

        wandb.watch(resnet_model, resnet_trainer.loss_fn, log="all", log_freq=10)
        name = "resnet_model exp2"
        wandb.define_metric("{name} train_loss", summary="min")
        wandb.define_metric("{name} valid_loss", summary="min")
        wandb.define_metric("{name} train_f1", summary="max")
        wandb.define_metric("{name} valid_f1", summary="max")

        resnet_trainer.train()

        ##############################################################################
        # evaluate the ResNet model on the test set
        ##############################################################################
        # load the model
        resnet_model = load_model(
            resnet_model,
            f"{config['checkpoints']}/resnet_model exp2.pt",
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=generated_images_loader,
            val_loader=val_loader,
            classes=classes,
        )

        f1_scores, avg_f1 = resnet_trainer.run(
            mode="test", dataloader=testloader, step="test step"
        )
        wandb.log({f"{name} avg_test_f1": avg_f1})
        wandb.log({f"{name} test_scores": f1_scores})

        ##############################################################################
        # exp3: train resnet model on 50% of the generated images and 50% of the real images
        ##############################################################################
        # instantiate model again
        resnet_model = create_model("ResNet", config.data["dataset"], num_classes)

        # take 50% of the generated images and 50% of the real images
        real_images, _ = random_split(
            trainset2, [math.ceil(len(trainset2) / 2), int(len(trainset2) / 2)]
        )
        fake_images, _ = random_split(
            generated_images,
            [math.ceil(len(generated_images) / 2), int(len(generated_images) / 2)],
        )

        # create a torch dataset from the real and fake images
        trainset3 = torch.utils.data.ConcatDataset([real_images, fake_images])
        trainset3_loader = DataLoader(
            trainset3,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # train the ResNet model
        # evaluate it on the 10% part of the second part
        # val dataloader
        val_loader = DataLoader(
            valset2, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset3_loader,
            val_loader=val_loader,
            classes=classes,
        )
        resnet_trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/resnet_model exp3.pt",
        )
        # overwrite default loss function
        resnet_trainer.loss_fn = f.cross_entropy

        wandb.watch(resnet_model, resnet_trainer.loss_fn, log="all", log_freq=10)
        name = "resnet_model exp3"
        wandb.define_metric("{name} train_loss", summary="min")
        wandb.define_metric("{name} valid_loss", summary="min")
        wandb.define_metric("{name} train_f1", summary="max")
        wandb.define_metric("{name} valid_f1", summary="max")

        resnet_trainer.train()

        ##############################################################################
        # evaluate the ResNet model on the test set
        ##############################################################################
        # load the model
        resnet_model = load_model(
            resnet_model,
            f"{config['checkpoints']}/resnet_model exp3.pt",
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset3_loader,
            val_loader=val_loader,
            classes=classes,
        )

        f1_scores, avg_f1 = resnet_trainer.run(
            mode="test", dataloader=testloader, step="test step"
        )
        wandb.log({f"{name} avg_test_f1": avg_f1})
        wandb.log({f"{name} test_scores": f1_scores})

        ##############################################################################
        # exp4: train resnet model on 90% of the generated images and 10% of the real images
        ##############################################################################
        # instantiate model again
        resnet_model = create_model("ResNet", config.data["dataset"], num_classes)

        # take 90% of the generated images and 10% of the real images
        real_images, _ = random_split(
            trainset2, [math.ceil(len(trainset2) / 10), int(len(trainset2) / 10 * 9)]
        )
        fake_images, _ = random_split(
            generated_images,
            [
                math.ceil(len(generated_images) / 10 * 9),
                int(len(generated_images) / 10),
            ],
        )

        # take 90% of the generated images and 10% of the real images
        trainset4 = torch.utils.data.ConcatDataset([real_images, fake_images])
        trainset4_loader = DataLoader(
            trainset4,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # train the ResNet model
        # evaluate it on the 10% part of the second part
        # val dataloader
        val_loader = DataLoader(
            valset2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset4_loader,
            val_loader=val_loader,
            classes=classes,
        )
        resnet_trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/resnet_model exp4.pt",
        )

        # overwrite default loss function
        resnet_trainer.loss_fn = f.cross_entropy

        wandb.watch(resnet_model, resnet_trainer.loss_fn, log="all", log_freq=10)
        name = "resnet_model exp4"
        wandb.define_metric("{name} train_loss", summary="min")
        wandb.define_metric("{name} valid_loss", summary="min")
        wandb.define_metric("{name} train_f1", summary="max")
        wandb.define_metric("{name} valid_f1", summary="max")

        resnet_trainer.train()

        ##############################################################################
        # evaluate the ResNet model on the test set
        ##############################################################################
        # load the model
        resnet_model = load_model(
            resnet_model,
            f"{config['checkpoints']}/resnet_model exp4.pt",
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset4_loader,
            val_loader=val_loader,
            classes=classes,
        )

        f1_scores, avg_f1 = resnet_trainer.run(
            mode="test", dataloader=testloader, step="test step"
        )
        wandb.log({f"{name} avg_test_f1": avg_f1})
        wandb.log({f"{name} test_scores": f1_scores})

        ##############################################################################
        # exp5: train resnet model on 10% of the generated images and 90% of the real images
        ##############################################################################
        # instantiate model again
        resnet_model = create_model("ResNet", config.data["dataset"], num_classes)

        # take 10% of the generated images and 90% of the real images
        real_images, _ = random_split(
            trainset2, [math.ceil(len(trainset2) / 10 * 9), int(len(trainset2) / 10)]
        )
        fake_images, _ = random_split(
            generated_images,
            [
                math.ceil(len(generated_images) / 10),
                int(len(generated_images) / 10 * 9),
            ],
        )

        # take 10% of the generated images and 90% of the real images
        trainset5 = torch.utils.data.ConcatDataset([real_images, fake_images])
        trainset5_loader = DataLoader(
            trainset5,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )

        # train the ResNet model
        # evaluate it on the 10% part of the second part
        # val dataloader
        val_loader = DataLoader(
            valset2,
            batch_size=batch_size,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset5_loader,
            val_loader=val_loader,
            classes=classes,
        )
        resnet_trainer.early_stopping = EarlyStopping(
            patience=config["early_stopping_patience"],
            verbose=True,
            path=f"{config['checkpoints']}/resnet_model exp5.pt",
        )
        # overwrite default loss function
        resnet_trainer.loss_fn = f.cross_entropy

        wandb.watch(resnet_model, resnet_trainer.loss_fn, log="all", log_freq=10)
        name = "resnet_model exp5"
        wandb.define_metric("{name} train_loss", summary="min")
        wandb.define_metric("{name} valid_loss", summary="min")
        wandb.define_metric("{name} train_f1", summary="max")
        wandb.define_metric("{name} valid_f1", summary="max")

        resnet_trainer.train()

        ##############################################################################
        # evaluate the ResNet model on the test set
        ##############################################################################
        # load the model
        resnet_model = load_model(
            resnet_model,
            f"{config['checkpoints']}/resnet_model exp5.pt",
        )
        resnet_trainer = ResNetTrainer(
            config=config,
            model=resnet_model,
            train_loader=trainset5_loader,
            val_loader=val_loader,
            classes=classes,
        )

        f1_scores, avg_f1 = resnet_trainer.run(
            mode="test", dataloader=testloader, step="test step"
        )

        wandb.log({f"{name} avg_test_f1": avg_f1})
        wandb.log({f"{name} test_scores": f1_scores})

        ##############################################################################
        # Done
        ##############################################################################
        wandb.finish()


if __name__ == "__main__":
    ##############################################################################
    # Load a config file
    ##############################################################################
    config_file = sys.argv[1]
    configurations = yaml.safe_load(open(config_file, "r"))

    config = Config(**configurations)
    main(config=config)
