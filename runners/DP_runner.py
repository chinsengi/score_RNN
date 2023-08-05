import torch
from utility import *
from models import rand_RNN, NeuralDyn
import logging
try:
    import tensorboardX
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
from data import UniformData, GMMData
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, LogNorm
import scienceplots
import torch.nn as nn

# plt.style.use('science')
__all__ = ['DP']

# double peak experiment
class DP():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.data_mean = torch.tensor([-1., 1.]).reshape([2, 1])
        self.data_std = torch.tensor([.25, .5]).reshape([2, 1])
        # self.data_mean = torch.tensor([-1., 1.]).reshape([2, 1])
        # self.data_std = torch.tensor([.5, .5]).reshape([2, 1])
        self.noise_start = 1
        self.noise_end = 1/50
        # torch.set_float32_matmul_precision('high')

    def train(self):
        out_dim, hid_dim = 1, self.args.hid_dim
        training_batch_size = 32

        GMM_mean = self.data_mean.to(self.device)
        GMM_std = self.data_std.to(self.device)
        data = GMMData(GMM_mean, GMM_std, n=10000)
        train_loader = torch.utils.data.DataLoader(data, batch_size= training_batch_size)

        # choosing the model
        logging.info("model used is :"+ self.args.model)
        model = self.set_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=.001, amsgrad=True)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

        # annealing noise
        n_level = self.args.noise_level
        noise_levels = [self.noise_start/math.exp(math.log(self.noise_start/self.noise_end)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{self.args.model}_ep{epoch}.pth")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001, amsgrad=True)
            for batchId, h in enumerate(train_loader):
                h = h.to(self.device)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{self.args.model}_ep{nepoch}.pth")
        save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{self.args.model}_chkpt{self.args.run_id}.pth")    

    def test(self):
        hid_dim, out_dim= self.args.hid_dim, 1
        samples = 0
        nsample = 5000
        if self.args.model=="SR":
            samples = (torch.rand([nsample, hid_dim])*4-2).to(self.device)
        elif self.args.model=="SO_SC" or self.args.model=="SO_FR":
            samples = (torch.rand([nsample, out_dim])*4-2).to(self.device)
        
        n_level = self.args.noise_level
        noise_levels = [self.noise_start/math.exp(math.log(self.noise_start/self.noise_end)*n/n_level) for n in range(n_level)]
        norm = LogNorm(vmin=min(noise_levels), vmax=max(noise_levels))
        colors = create_color_gradient(noise_levels, norm)

        model = self.set_model()
        with torch.no_grad():
            # plot score function during training
            lower_bound = torch.min(self.data_mean) - torch.max(self.data_std)*3
            upper_bound = torch.max(self.data_mean) + torch.max(self.data_std)*3
            x_range = torch.arange(lower_bound, upper_bound, .1)
            for i in range(1,n_level+1):
                load(f"./model/DP/{self.args.run_id}/{self.args.model}_ep{i*(self.args.nepochs//self.args.noise_level)}.pth", model)
                model.set_weight()
                tmp = model.score(x_range.to(self.device).reshape(1, -1, 1)).squeeze().detach().cpu().numpy()
                plt.plot(x_range, tmp, color=colors[i-1])
            true_score = score_GMM_1D(x_range, self.data_mean, self.data_std**2)
            plt.plot(x_range, true_score, color="orange", label="true score")
            plt.legend()
            sm = ScalarMappable(norm=norm, cmap="Blues_r")
            sm.set_array([])
            plt.colorbar(sm)
            plt.tight_layout()
            savefig(path="./image/DP", filename=self.args.model+"_score_func")
            
            # get samples
            load(f"./model/DP/{self.args.run_id}/{self.args.model}_chkpt{self.args.run_id}.pth", model)
            model.set_weight()
            model.dt = 1e-4
            samples = gen_sample(model, samples, 10000)
            plt.figure()
            samples = model.W_out(samples)
            samples = samples.detach().cpu().numpy()
            logging.info(samples.shape)
            bin_edges = x_range.numpy()
            _, bins, _ = plt.hist(samples, bins=bin_edges, label=  "sampled points")
            plt.plot(x_range, mixture_pdf(x_range, self.data_mean, self.data_std**2)*nsample*(bins[2]-bins[1]), label="Scaled density function", color="orange")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            savefig(path="./image/DP", filename=self.args.model+"_DP_sampled")

    def set_model(self):
        # choosing the nonlinearity
        if self.args.nonlin == "tanh":
            nonlin = nn.Tanh()
        elif self.args.nonlin == "relu":
            nonlin = nn.ReLU()
        else:
            nonlin = nn.Softplus()

        if self.args.model == "SR":
            print("Using reservoir-sampler arch")
            model = rand_RNN(self.args.hid_dim, 1, non_lin=nonlin)
        elif self.args.model == "SO_SC":
            print("Using sampler-only arch with synaptic current dynamics")
            model = NeuralDyn(1, non_lin=nonlin)
        elif self.args.model == "SO_FR":
            print("Using sampler-only arch with firing rate dynamics")
            model = NeuralDyn(1, synap=False, non_lin=nonlin)
        else:
            return None

        return model.to(self.device)