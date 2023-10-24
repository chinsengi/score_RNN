import os
import sys

sys.path.insert(0, os.path.abspath("../../."))
import argparse
from tqdm import tqdm
import torch
from models import Autoencoder
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utility import create_dir, plot_true_and_recon_img, use_gpu


# save to tensorboard
# Hyperparameters no cmd for now
arg = argparse.Namespace()
arg.hidden_dim = 32
# arg.input_dim = 256
# arg.r_lr = 1
# arg.lmda = 0.001
# arg.maxiter = 500

arg.dataset_name = "cifar10"
arg.img_size = 32
arg.n_channel = 3
arg.batch_size = 64
arg.learning_rate = 0.0005
arg.epoch = 250

hyperparam_str = f"ae-hidden_dim-{arg.hidden_dim}-lr-{arg.learning_rate}"
train_board = SummaryWriter(f"run/ae-{hyperparam_str}-train")
test_board = SummaryWriter(f"run/ae-{hyperparam_str}-test")

# if use cuda
device = use_gpu()
# create net
autoencoder = Autoencoder(arg.hidden_dim, arg.img_size, arg.n_channel).to(device)


def create_dataloader(batch_size, train=True, dataset_name="MNIST"):
    dataset = eval(
        f"torchvision.datasets.{dataset_name.upper()}('./data/', train=train, download=True)"
    )
    if "numpy" in str(dataset.data.__class__):
        data = torch.from_numpy(dataset.data)
    else:
        data = dataset.data
    train_data = data.to(torch.float32).reshape(len(dataset), -1)
    train_data = train_data / 255
    train_data = (train_data - 0.1307) / 0.3081

    return DataLoader(
        torch.utils.data.TensorDataset(train_data), batch_size=batch_size, shuffle=True
    )


# dataloader
train_dataloader = create_dataloader(
    arg.batch_size, train=True, dataset_name=arg.dataset_name
)
test_dataloader = create_dataloader(
    arg.batch_size, train=False, dataset_name=arg.dataset_name
)

# create ckpt folder
ckpt_dir = "model/ae"
create_dir(ckpt_dir)
# train
optim = torch.optim.AdamW(autoencoder.parameters(), lr=arg.learning_rate)
for e in range(arg.epoch):
    running_loss_train = 0
    c = 0
    autoencoder.train()
    for img_batch in tqdm(
        train_dataloader,
        desc="training",
        total=len(train_dataloader),
        dynamic_ncols=True,
    ):
        img_batch = img_batch[0]
        img_batch = img_batch.reshape(
            img_batch.shape[0], arg.n_channel, arg.img_size, arg.img_size
        ).to(device)
        # update
        pred = autoencoder(img_batch)
        loss = torch.pow(pred - img_batch, 2).sum(1).mean()
        running_loss_train += loss.item()
        loss.backward()
        # update U
        optim.step()
        # zero grad
        optim.zero_grad()
        c += 1
    train_board.add_scalar("Loss", running_loss_train / c, e)

    running_loss_test = 0
    c = 0
    autoencoder.eval()
    for img_batch in tqdm(test_dataloader, desc="testing", total=len(test_dataloader)):
        img_batch = img_batch[0]
        img_batch = img_batch.reshape(
            img_batch.shape[0], arg.n_channel, arg.img_size, arg.img_size
        ).to(device)
        # update
        pred = autoencoder(img_batch)
        loss = torch.pow(pred - img_batch, 2).sum(1).mean()
        running_loss_test += loss.item()
        c += 1
    test_board.add_scalar("Loss", running_loss_test / c, e)

    # plotting
    fig, ax = plot_true_and_recon_img(
        img_batch[0]
        .reshape(arg.img_size, arg.img_size, arg.n_channel)
        .detach()
        .cpu()
        .numpy(),
        pred[0]
        .reshape(arg.img_size, arg.img_size, arg.n_channel)
        .detach()
        .cpu()
        .numpy(),
    )
    test_board.add_figure("Recon", fig, global_step=e)
    if e % 10 == 9:
        # save checkpoint
        torch.save(
            autoencoder.state_dict(), f"{ckpt_dir}/ae-{hyperparam_str}-ckpt-{e+1}.pth"
        )

torch.save(autoencoder.state_dict(), f"{ckpt_dir}/ae-{hyperparam_str}-ckpt-{e+1}.pth")
