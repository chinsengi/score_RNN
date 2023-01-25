import torch
from utility import *
from models import FR
from data import UniformData, GMMData


if __name__ == "__main__":
    device = use_gpu()
    hid_dim = 16
    model = FR(hid_dim).to(device)
    mean = torch.zeros(hid_dim-1, device=device)
    var  = torch.ones(hid_dim-1, device=device)/2
    GMM_mean = torch.tensor([-1, 1], device=device).unsqueeze(0)
    GMM_var = torch.tensor([0.5, .5], device=device).unsqueeze(0)
    # load("./model/best_model_ep200", model)
    # data = UniformData(lower = -2., upper = 2., n=10000, ndim=hid_dim)
    data = GMMData(GMM_mean, GMM_var, mean, var, n=10000)
    train_loader = torch.utils.data.DataLoader(data, batch_size= 256)
    # train_normal(model, train_loader, device)
    train_GMM_cn(model, train_loader, GMM_mean, GMM_var, device)