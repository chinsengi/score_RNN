import torch
from utility import *
from models import rand_RNN
from data import UniformData, GMMData
import torchvision
import numpy as np
import matplotlib.pyplot as plt
# from sklearnex import patch_sklearn
# patch_sklearn()

from sklearn.decomposition import PCA

if __name__ == "__main__":
    create_dir("./model")
    create_dir("./image")
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
        out_dim, hid_dim = 300, 5012
        train_batch_size =  64# Define train batch size

        #MNIST data_matrix used for PCA
        train_data = torchvision.datasets.MNIST('./data/', train=True, download=True)
        trans = torchvision.transforms.Normalize(.1307, .3081)
        train_data = train_data.data.to(torch.float32).reshape(len(train_data), -1)
        train_data = torch.nn.functional.normalize(train_data, dim = 1)
        plt.imshow(train_data[0].reshape([28,28]))
        plt.savefig('./image/digit.png')
        pca = PCA(n_components=out_dim)
        hidden_states = pca.fit_transform(train_data)
        recovered_hid = pca.inverse_transform(hidden_states)
        plt.imshow(recovered_hid[0].reshape([28,28]))
        plt.savefig('./image/digit_recovered.png')
        print(hidden_states.shape)

        train_dataset = torch.utils.data.TensorDataset(torch.tensor(hidden_states))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        
        model = rand_RNN(hid_dim, out_dim).to(device)
        # train_MNIST(model, train_loader, device)

        load(f"./model/{model.__class__.__name__}_MNIST_ep400", model)

        model.set_weight()
        samples = (torch.rand([10, hid_dim])*20-10).to(device)
        with torch.no_grad():
            model.dt = 5e-2
            samples = gen_sample(model, samples, 5000)
            samples = model.W_out(samples).detach().cpu().numpy()
            samples = pca.inverse_transform(samples).reshape(len(samples), 28, 28)
            print(samples.shape)
            fig, axes = plt.subplots(2, 5)
            for i in range(2):
                for j in range(5):
                    ax = axes[i,j]
                    ax.imshow(samples[i*5+j])
            plt.savefig("./image/digit_sampled.png")