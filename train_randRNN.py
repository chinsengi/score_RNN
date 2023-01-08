import torch
from utility import *
from models import rand_RNN
from data import UniformData, GMMData


if __name__ == "__main__":
    device = use_gpu()
    out_dim, hid_dim = 1,128
    model = rand_RNN(hid_dim, out_dim).to(device)
    mean = torch.zeros(out_dim-1, device=device)
    var  = torch.ones(out_dim-1, device=device)/2
    GMM_mean = torch.tensor([-2, 2], device=device).unsqueeze(0)
    GMM_var = torch.tensor([0.5, .5], device=device).unsqueeze(0)
    data = GMMData(GMM_mean, GMM_var, mean, var, n=10000)
    train_loader = torch.utils.data.DataLoader(data, batch_size= 128)
    # train_normal(model, train_loader, device)
    train_GMM_cn(model, train_loader, GMM_mean, GMM_var, device)