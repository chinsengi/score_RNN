import torch
from utility import *
from models import rand_RNN, NeuralDyn
import logging
try:
    import tensorboardX
except ModuleNotFoundError:
    pass
import matplotlib.pyplot as plt
import scienceplots
plt.style.use('science')
plt.rcParams.update({
    "text.usetex": False,
})
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['axes.labelsize'] = 16

from data import LAPData
import numpy as np
from scipy.stats import kurtosis
from matplotlib.colors import ListedColormap
# import scienceplots

# plt.style.use('science')
__all__ = ['LAP']

# double peak experiment
class LAP():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        # torch.set_float32_matmul_precision('high')

    def train(self):
        out_dim, hid_dim = 2, self.args.hid_dim
        training_batch_size = 128

        LAP_mean = torch.tensor([[0., 0.], [0., 0.]])
        LAP_std = torch.tensor([
            [[1, 0.9], [0.9, 1.]],
            [[1, -0.9], [-0.9, 1]],
        ])

        data = LAPData(LAP_mean, LAP_std, n=20000)
        train_loader = torch.utils.data.DataLoader(data, batch_size= training_batch_size)

        # choosing the model
        print("model used is :"+ self.args.model)
        model = self.set_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.00005)

        # annealing noise
        n_level = self.args.noise_level
        noise_levels = [1/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/LAP/{self.args.run_id}", f"{self.args.model}_ep{epoch}.pth")
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
            for batchId, h in enumerate(train_loader):
                h = h.to(self.device, dtype=torch.float32)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(model, optimizer, f"./model/LAP/{self.args.run_id}", f"{self.args.model}_ep{nepoch}.pth")
        save(model, optimizer, f"./model/LAP/{self.args.run_id}", f"{self.args.model}_chkpt{self.args.run_id}.pth")    

    def test(self):
        hid_dim, out_dim = self.args.hid_dim, 2
        samples = 0
        nsample = 10000
        if self.args.model=="SR":
            #samples = (torch.rand([nsample, hid_dim])*4-2).to(self.device)
            samples = (torch.randn([nsample, hid_dim])).to(self.device)
        elif self.args.model=="SO_SC" or self.args.model=="SO_FR":
            samples = (torch.rand([nsample, out_dim])*4-2).to(self.device)
        
        with torch.no_grad():
            # true samples
            LAP_mean = torch.tensor([[0., 0.], [0., 0.]])
            LAP_std = torch.tensor([
                [[1, 0.9], [0.9, 1.]],
                [[1, -0.9], [-0.9, 1]],
            ])
            data = LAPData(LAP_mean, LAP_std, n=nsample)
            train_loader = torch.utils.data.DataLoader(data, batch_size=nsample)
            true_samples = next(iter(train_loader)).detach().cpu().numpy()
            # learned samples 
            model = self.set_model()
            load(f"./model/LAP/{self.args.run_id}/{self.args.model}_chkpt{self.args.run_id}.pth", model)
            model.set_weight()
            model.dt = 1e-4
            samples = gen_sample(model, samples, 20000)
            samples = model.W_out(samples)
            samples = samples.detach().cpu().numpy()
            logging.info(samples.shape)
            # side by side figure -- true samples vs generated samples, and krutosis group bar plot
            fig, axes = plt.subplots(1,2, figsize=(10,5), sharex=True, sharey=True)
            # plot true sample historgram
            axes[0].hist2d(true_samples[:,0], true_samples[:,1], bins=[np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1)], label="true samples")
            # plot learned sampled historgram
            axes[1].hist2d(samples[:,0], samples[:,1], bins=[np.arange(-6, 6, 0.1), np.arange(-6, 6, 0.1)], label="generated samples")
            plt.xlim(-5, 5)
            plt.ylim(-5, 5)
            savefig(path="./image/LAP", filename=self.args.model+"_LAP_true_v_gen_density", format="png")
             
            # compare kurtosis
            plt.figure(figsize=(5, 5))
            true_kurts = [kurtosis(true_samples[:,0]), kurtosis(true_samples[:,1])]
            gen_kurts = [kurtosis(samples[:,0]), kurtosis(samples[:,1])]
            categories = ['True sample', 'Generated sample']
            group_names = ['x', 'y']
            values = np.array([true_kurts, gen_kurts])

            # Set the position of the bars on the x-axis
            bar_width = 0.2
            x = np.arange(len(group_names))

            # Plot the bars for each category
            for i in range(len(categories)):
                plt.bar(x + i * bar_width, values[i], width=bar_width, label=categories[i])

            # Set the x-axis labels, y-axis label, and the plot title
            plt.ylabel('Kurtosis')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2, fontsize=16)
        
            # Set the tick labels on the x-axis
            plt.xticks(x + bar_width * (len(categories) - 1) / 2, ["", ""])
            savefig(path="./image/LAP", filename=self.args.model+"_LAP_true_v_gen_kurt", format="png")


    def set_model(self):
        if self.args.model == "SR":
            print("Using reservoir-sampler arch")
            model = rand_RNN(self.args.hid_dim, 2)
        elif self.args.model == "SO_SC":
            print("Using sampler-only arch with synaptic current dynamics")
            model = NeuralDyn(2)
        elif self.args.model == "SO_FR":
            print("Using sampler-only arch with firing rate dynamics")
            model = NeuralDyn(2, synap=False)
        else:
            return None

        return model.to(self.device)