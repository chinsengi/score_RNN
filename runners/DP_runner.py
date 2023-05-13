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
from matplotlib.colors import ListedColormap
# import scienceplots

# plt.style.use('science')
__all__ = ['DP']

# double peak experiment
class DP():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.n_level = 20
        # torch.set_float32_matmul_precision('high')

    def train(self):
        out_dim, hid_dim = 1, self.args.hid_dim
        training_batch_size = 128

        GMM_mean = torch.tensor([-1., 1.], device=self.device).reshape([2, 1])
        GMM_std = torch.tensor([.5]).unsqueeze(0).repeat(2, 1).to(GMM_mean)
        data = GMMData(GMM_mean, GMM_std, n=10000)
        train_loader = torch.utils.data.DataLoader(data, batch_size= training_batch_size)

        # choosing the model
        print("model used is :"+ self.args.model)
        model = self.set_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)

        # annealing noise
        n_level = self.n_level
        noise_levels = [1/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{self.args.model}_ep{epoch}.pth")
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
            for batchId, h in enumerate(train_loader):
                h = h.to(self.device)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(model, optimizer, f"./model/DP/{self.args.run_id}", f"{self.args.model}_chkpt{self.args.run_id}.pth")    

    def test(self):
        hid_dim, out_dim= self.args.hid_dim, 1
        samples = 0
        nsample = 5000
        if self.args.model=="SR":
            samples = (torch.rand([nsample, hid_dim])*4-2).to(self.device)
        elif self.args.model=="SO_SC" or self.args.model=="SO_FR":
            samples = (torch.rand([nsample, out_dim])*4-2).to(self.device)
        
        n_level = 20
        colors = [(1.0, 1.0, 1.0), (0.9, 0.9, 1.0), (0.8, 0.8, 1.0), (0.7, 0.7, 1.0), 
             (0.6, 0.6, 1.0), (0.5, 0.5, 1.0), (0.4, 0.4, 1.0), (0.3, 0.3, 1.0), 
             (0.2, 0.2, 1.0), (0.1, 0.1, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 0.9), 
             (0.0, 0.0, 0.8), (0.0, 0.0, 0.7), (0.0, 0.0, 0.6), (0.0, 0.0, 0.5), 
             (0.0, 0.0, 0.4), (0.0, 0.0, 0.3), (0.0, 0.0, 0.2), (0.0, 0.0, 0.1)]
        cmap = ListedColormap(colors)
        with torch.no_grad():
            for i in range(1,n_level):
                model = self.set_model()
                load(f"./model/DP/{self.args.run_id}/{self.args.model}_ep{i*(self.args.nepochs//self.n_level)}.pth", model)
                model.set_weight()
                tmp = model.score(torch.arange(-5, 5, .1).to(self.device).reshape(1, -1, 1)).squeeze().detach().cpu().numpy()
                plt.plot(np.arange(-5,5,.1), tmp, color=colors[i])
            true_score = score_GMM_1D(torch.arange(-5,5,.1), torch.tensor([-1,1]), torch.tensor([0.5**2, 0.5**2]))
            plt.plot(np.arange(-5, 5, .1), true_score, color="red", label="true score")
            plt.legend(fontsize=12)
            savefig(path="./image/DP", filename=self.args.model+"_score_func")
            model = self.set_model()
            load(f"./model/DP/{self.args.run_id}/{self.args.model}_chkpt{self.args.run_id}.pth", model)
            model.set_weight()
            model.dt = 1e-3
            samples = gen_sample(model, samples, 5000)
            plt.figure()
            samples = model.W_out(samples)
            samples = samples.detach().cpu().numpy()
            logging.info(samples.shape)
            _, bins, _ = plt.hist(samples, bins=100, label=  "sampled points")
            plt.plot(np.arange(-3,3,.1), mixture_pdf(torch.arange(-3,3,.1), torch.tensor([-1,1]), torch.tensor([0.5**2,0.5**2]))*nsample*(bins[1]-bins[0]), label="Scaled density function")
            plt.legend(fontsize=12)
            savefig(path="./image/DP", filename=self.args.model+"_DP_sampled")

    def set_model(self):
        if self.args.model == "SR":
            print("Using reservoir-sampler arch")
            model = rand_RNN(self.args.hid_dim, 1)
        elif self.args.model == "SO_SC":
            print("Using sampler-only arch with synaptic current dynamics")
            model = NeuralDyn(1)
        elif self.args.model == "SO_FR":
            print("Using sampler-only arch with firing rate dynamics")
            model = NeuralDyn(1, synap=False)
        else:
            return None

        return model.to(self.device)