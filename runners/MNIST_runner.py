import torch
from utility import *
from models import rand_RNN
import torchvision
import logging
import tensorboardX
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
from sklearnex import patch_sklearn
patch_sklearn()

__all__ = ['MNIST']

class MNIST():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.out_dim, self.hid_dim = 300, args.hid_dim
        self.train_batch_size =  64 # Define train batch size

        #MNIST data_matrix used for PCA
        train_data = torchvision.datasets.MNIST('./data/', train=True, download=True)
        train_data = train_data.data.to(torch.float32).reshape(len(train_data), -1)
        train_data = torch.nn.functional.normalize(train_data, dim = 1)
        plt.imshow(train_data[0].reshape([28,28]))
        savefig(path="./image/MNIST", filename="_digit.png")
        self.pca = PCA(n_components=self.out_dim)
        self.hidden_states = self.pca.fit_transform(train_data)*100
        recovered_hid = self.pca.inverse_transform(self.hidden_states[0]/100)
        plt.imshow(recovered_hid.reshape([28,28]))
        savefig(path="./image/MNIST", filename="_digit_recovered.png")
        logging.info(self.hidden_states.shape)

    def train(self):
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.hidden_states))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        
        model = rand_RNN(self.hid_dim, self.out_dim).to(self.device)
        # annealing noise
        n_level = 10
        noise_levels = [1/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        if self.args.resume:
            load(f"./model/{self.args.run_id}/{model.__class__.__name__}_MNIST_chkpt{self.args.run_id}", model, optimizer)
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/MNIST/{self.args.run_id}", f"{model.__class__.__name__}_MNIST_ep{epoch}")
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
            for h in train_loader:
                # print(batchId)
                h = h[0].to(self.device, torch.float32)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(model, optimizer, f"./model/MNIST/", f"{model.__class__.__name__}_MNIST_chkpt{self.args.run_id}")


    def test(self):
        model = rand_RNN(self.hid_dim, self.out_dim).to(self.device)

        load(f"./model/MNIST/{model.__class__.__name__}_MNIST_chkpt{self.args.run_id}", model)
        # load(f"./model/MNIST/{self.args.run_id}/{model.__class__.__name__}_MNIST_ep{360}", model)

        model.set_weight()
        samples = (torch.rand([10, self.hid_dim])*10-5).to(self.device)
        # samples = (torch.randn([10, self.out_dim])).to(self.device)
        with torch.no_grad():
            model.dt = 1e-3
            samples = gen_sample(model, samples, 10000)
            samples = model.W_out(samples)
            samples = samples.detach().cpu().numpy()/100
            samples = self.pca.inverse_transform(samples).reshape(len(samples), 28, 28)
            print(samples.shape)
            fig, axes = plt.subplots(2, 5)
            for i in range(2):
                for j in range(5):
                    ax = axes[i,j]
                    ax.imshow(samples[i*5+j])
            savefig(path="./image/MNIST", filename="_digit_sampled.png")
