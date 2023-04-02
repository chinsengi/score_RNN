import torch
from utility import *
from models import rand_RNN, FR
import logging
try:
    import tensorboardX
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
from data import UniformData, GMMData
import numpy as np
from matplotlib.colors import ListedColormap

__all__ = ['DP']

class DP():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        torch.set_float32_matmul_precision('high')

    def train(self):
        out_dim, hid_dim = 1, self.args.hid_dim
        training_batch_size = 256

        GMM_mean = torch.tensor([-1., 1.], device=self.device).reshape([2, 1])
        GMM_std = torch.tensor([.5]).unsqueeze(0).repeat(2, 1).to(GMM_mean)
        data = GMMData(GMM_mean, GMM_std, n=10000)
        train_loader = torch.utils.data.DataLoader(data, batch_size= training_batch_size)
        model = rand_RNN(hid_dim, out_dim).to(self.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)

        # annealing noise
        n_level = 20
        noise_levels = [10/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{model.__class__.__name__}_ep{epoch}.pth")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
            for batchId, h in enumerate(train_loader):
                h = h.to(self.device)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{model.__class__.__name__}_chkpt{self.args.run_id}.pth")    

    def test(self):
        hid_dim, out_dim= self.args.hid_dim, 1
        samples = (torch.rand([1000, hid_dim])*4-2).to(self.device)
        n_level = 20
        colors = [(1.0, 1.0, 1.0), (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), 
             (0.6, 0.6, 1.0), (0.5, 0.5, 1.0), (0.4, 0.4, 1.0), (0.3, 0.3, 1.0), 
             (0.2, 0.2, 1.0), (0.1, 0.1, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.9), 
             (0.0, 0.0, 0.8), (0.0, 0.0, 0.7), (0.0, 0.0, 0.6), (0.0, 0.0, 0.5), 
             (0.0, 0.0, 0.4), (0.0, 0.0, 0.3), (0.0, 0.0, 0.2), (0.0, 0.0, 0.1)]
        cmap = ListedColormap(colors)
        with torch.no_grad():
            for i in range(1,n_level):
                model = rand_RNN(hid_dim, out_dim).to(self.device)
                load(f"./model/DP/{self.args.run_id}/{model.__class__.__name__}_ep{i*20}.pth", model)
                model.set_weight()
                tmp = model.score(torch.arange(-5, 5, .1).to(self.device).reshape(1, -1, 1)).squeeze().detach().cpu().numpy()
                plt.plot(np.arange(-5,5,.1), tmp, color=colors[i])
            true_score = score_GMM_1D(torch.arange(-5,5,.1), torch.tensor([-1,1]), torch.tensor([0.5**2, 0.5**2]))
            plt.plot(np.arange(-5, 5, .1), true_score, color="red", label="true score")
            plt.legend()
            savefig(path="./image/DP", filename="_score_func.png")
            model = rand_RNN(hid_dim, out_dim).to(self.device)
            load(f"./model/DP/{self.args.run_id}/{model.__class__.__name__}_chkpt{self.args.run_id}.pth", model)
            model.set_weight()
            model.dt = 1e-2
            # samples = gen_sample(model, samples, 10000)
        # plt.figure()
        # samples = model.W_out(samples)
        # samples = samples.detach().cpu().numpy()
        # logging.info(samples.shape)
        # _, bins, _ = plt.hist(samples, bins=40, label="sampled points")
        # plt.plot(np.arange(-3,3,.1), mixture_pdf(torch.arange(-3,3,.1), torch.tensor([-1,1]), torch.tensor([0.5**2,0.5**2]))*1000*(bins[1]-bins[0]), label="Scaled density function")
        # plt.legend()
        # savefig(path="./image/DP", filename="_DP_sampled.png")

