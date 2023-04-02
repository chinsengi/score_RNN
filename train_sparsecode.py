import os
import sys
sys.path.insert(0, os.path.abspath('../../.'))
import argparse
from tqdm import tqdm
import torch
from models import SparseNet
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision
from utility import create_dir, plot_spatial_rf, plot_true_and_recon_img


# save to tensorboard
# Hyperparameters no cmd for now
arg = argparse.Namespace()
arg.input_dim = 28 * 28
arg.hidden_dim = arg.input_dim * 4
arg.r_lr = 0.75
arg.lmda = 0.001
arg.maxiter = 500

arg.batch_size = 500
arg.learning_rate = 0.05
arg.epoch = 200

hyperparam_str = f"hidden_dim-{arg.hidden_dim}-r_lr-{arg.r_lr}-lmda-{arg.lmda}-lr-{arg.learning_rate}" 
train_board = SummaryWriter(f"run/sparse-net-{hyperparam_str}-train")
test_board = SummaryWriter(f"run/sparse-net-{hyperparam_str}-test")

# if use cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# create net
sparse_net = SparseNet(arg.input_dim, arg.hidden_dim, arg.r_lr, arg.lmda, arg.maxiter, device).to(device)
# load data
train_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ])),
  batch_size=arg.batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./data/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.1307,), (0.3081,))
                             ])),
  batch_size=arg.batch_size, shuffle=True)

# create ckpt folder
ckpt_dir = "model/sparse" 
create_dir(ckpt_dir)
# train
optim = torch.optim.SGD([{'params': sparse_net.U.weight, "lr": arg.learning_rate}])
for e in range(arg.epoch):
    running_loss_train = 0
    c = 0
    for img_batch, _ in tqdm(train_dataloader, desc='training', total=len(train_dataloader)):
        img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
        # update
        pred = sparse_net(img_batch)
        loss = torch.pow(pred - img_batch, 2).sum(1).mean()
        running_loss_train += loss.item()
        loss.backward()
        # update U
        optim.step()
        # zero grad
        optim.zero_grad()
        # norm
        sparse_net.normalize_weights()
        c += 1
    train_board.add_scalar('Loss', running_loss_train / c, e)

    running_loss_test = 0
    c = 0
    for img_batch, _ in tqdm(test_dataloader, desc='testing', total=len(test_dataloader)):
        img_batch = img_batch.reshape(img_batch.shape[0], -1).to(device)
        # update
        pred = sparse_net(img_batch)
        loss = torch.pow(pred - img_batch, 2).sum(1).mean()
        running_loss_test += loss.item()
        c += 1
    test_board.add_scalar('Loss', running_loss_test / c, e)

    # plotting
    fig, ax = plot_spatial_rf(sparse_net.U.weight.T.data.reshape(sparse_net.hidden_dim, -1).detach().cpu().numpy()[:100])
    train_board.add_figure('RF', fig, global_step=e)
    fig, ax = plot_true_and_recon_img(img_batch[0].reshape(28, 28).detach().cpu().numpy(), pred[0].reshape(28, 28).detach().cpu().numpy())
    test_board.add_figure('Recon', fig, global_step=e)
    if e % 10 == 9:
        # save checkpoint
        torch.save(sparse_net, f"{ckpt_dir}/sparse-{hyperparam_str}-ckpt-{e+1}.pth")

torch.save(sparse_net, f"{ckpt_dir}/sparse-{hyperparam_str}-ckpt-{e+1}.pth")
