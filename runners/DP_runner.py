import torch
from utility import *
from models import rand_RNN, FR
import logging
import tensorboardX
import matplotlib.pyplot as plt
from data import UniformData, GMMData
import numpy as np

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
        GMM_var = torch.tensor([.5]).unsqueeze(0).repeat(2, 1).to(GMM_mean)
        data = GMMData(GMM_mean, GMM_var, n=10000)
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
        with torch.no_grad():
            for i in range(1,n_level):
                model = rand_RNN(hid_dim, out_dim).to(self.device)
                load(f"./model/DP/{self.args.run_id}/{model.__class__.__name__}_ep{i*20}.pth", model)
                model.set_weight()
                tmp = model.score(torch.arange(-5, 5, .1).to(self.device).reshape(1, -1, 1)).squeeze().detach().cpu().numpy()
                plt.plot(np.arange(-5,5,.1), tmp)
            savefig(path="./image/DP", filename="_score_func.png")
            model = rand_RNN(hid_dim, out_dim).to(self.device)
            load(f"./model/DP/{self.args.run_id}/{model.__class__.__name__}_chkpt{self.args.run_id}.pth", model)
            model.set_weight()
            model.dt = 1e-2
            samples = gen_sample(model, samples, 10000)
        plt.figure()
        samples = model.W1(samples).detach().cpu().numpy()
        # samples = samples.detach().cpu().numpy()
        plt.hist(samples, bins=40)
        savefig(path="./image/DP", filename="_DP_sampled.png")

