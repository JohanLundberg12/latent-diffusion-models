import torch
import torch.optim as optim
import torch.nn.functional as f
from torch import nn
import matplotlib
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

import time
from tqdm import tqdm
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models.SparseAutoEncoder import SparseAutoEncoder
from utils.data_utils import get_device

matplotlib.style.use("ggplot")


# for saving the reconstructed images
def save_decoded_image(img, name):
    img = img.view(img.size(0), 1, 28, 28)
    save_image(img, name)


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(
        f.sigmoid(rho_hat), 1
    )  # sigmoid because we need the probability distributions
    rho = torch.tensor([rho] * len(rho_hat)).to(device)

    return torch.sum(
        rho * torch.log(rho / rho_hat)
        + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat))
    )


# define the sparse loss function
def sparse_loss(rho, images):
    values = images
    loss = 0
    for i in range(len(model_children)):
        values = model_children[i](values)
        loss += kl_divergence(rho, values)
    return loss


def train(model, dataloader, epoch):
    model.train()
    running_loss = 0.0

    for i, data in enumerate(dataloader):
        img, targets = data
        img = img.to(device)
        optimizer.zero_grad()
        outputs = model(img)
        mse_loss = criterion(outputs, img)

        if ADD_SPARSITY:
            sparsity = sparse_loss(RHO, img)
            loss = mse_loss + BETA * sparsity
        else:
            loss = mse_loss

        loss.backward()
        optimizer.step()
        running_loss + loss.item()
    epoch_loss = running_loss / len(dataloader)

    print(f"Train Loss: {epoch_loss:.3f}")
    # save the reconstructed images
    save_decoded_image(outputs.cpu().data, f"./results/train{epoch}.png")

    return epoch_loss


# define the validation function
def validate(model, dataloader, epoch):
    print("Validating")
    model.eval()
    running_loss = 0.0
    with torch.no_grad():  # don't want to calculate gradients here
        for i, data in tqdm(
            enumerate(dataloader), total=int(len(testset) / dataloader.batch_size)
        ):
            img, targets = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            outputs = model(img)
            loss = criterion(outputs, img)
            running_loss += loss.item()

    epoch_loss = running_loss / len(testloader)

    print(f"Val Loss: {epoch_loss:.3f}")

    # save the reconstructed images
    outputs = outputs.view(outputs.size(0), 1, 28, 28).cpu().data
    save_image(outputs, f"../outputs/images/reconstruction{epoch}.png")

    return epoch_loss


if __name__ == "__main__":
    EPOCHS = 10
    BETA = 0.001
    ADD_SPARSITY = True
    RHO = 0.05
    LEARNING_RATE = 1e-4
    BATCH_SIZE = 32

    # image transformations
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )
    trainset = datasets.FashionMNIST(
        root="./data", train=True, download=True, transform=transform
    )
    testset = datasets.FashionMNIST(
        root="./data", train=False, download=True, transform=transform
    )

    # trainloader
    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
    # testloader
    testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    model = SparseAutoEncoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model_children = list(model.children())

    # train and validate the autoencoder neural network
    train_loss = []
    val_loss = []
    start = time.time()
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1} of {EPOCHS}")
        train_epoch_loss = train(model, trainloader, epoch)
        val_epoch_loss = validate(model, testloader, epoch)
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)
    end = time.time()
    print(f"{(end-start)/60:.3} minutes")
    # save the trained model
    torch.save(model.state_dict(), f"./models/sparse_ae{EPOCHS}.pth")

    # loss plots
    plt.figure(figsize=(10, 7))
    plt.plot(train_loss, color="orange", label="train loss")
    plt.plot(val_loss, color="red", label="validataion loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("./results/loss.png")
    plt.show()
