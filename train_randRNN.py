import torch
from utility import *
from models import rand_RNN
from data import UniformData, GMMData
import torchvision
import numpy as np


if __name__ == "__main__":
    device = use_gpu()
    data_type = "MNIST"
    if data_type == "GMM":
        out_dim, hid_dim = 1,512
        mean = torch.zeros(out_dim-1, device=device)
        var  = torch.ones(out_dim-1, device=device)/2
        GMM_mean = torch.tensor([-1, 1], device=device).unsqueeze(0)
        GMM_var = torch.tensor([0.5, .5], device=device).unsqueeze(0)
        data = GMMData(GMM_mean, GMM_var, mean, var, n=10000)
        train_loader = torch.utils.data.DataLoader(data, batch_size= 256)
        model = rand_RNN(hid_dim, out_dim).to(device)
        # train_normal(model, train_loader, device)
        train_GMM_cn(model, train_loader, GMM_mean, GMM_var, device)
    elif data_type == "MNIST":
        out_dim, hid_dim = 1,512
        train_batch_size =  64# Define train batch size

        # Use the following code to load and normalize the dataset
        train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./data/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=train_batch_size, shuffle=True)
        
        #data_matrix used for PCA
        train_data = np.vstack([x for x,_ in train_loader])
        print(train_data.shape)
        model = rand_RNN(hid_dim, out_dim).to(device)