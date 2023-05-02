import torch
from utility import *
from models import rand_RNN, SparseNet, FR
import torchvision
import logging

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import time
# from sklearnex import patch_sklearn
# patch_sklearn()

__all__ = ['MNIST']

class MNIST():
    

    class SparseCoding():
        def __init__(self, sparse_weight_path, device, feature_dim=3136, batch_size=500):
            # TODO: don't hard code this.
            self.device = device
            self.feature_dim = feature_dim
            self.net = SparseNet(784, feature_dim, 0.75, 0.001, 500, device).to(device)
            self.net.load_state_dict(torch.load(sparse_weight_path))
            self.net.eval()
            self.batch_size = batch_size

        def fit_transform(self, data):

            result = torch.zeros((data.size(0), self.feature_dim), requires_grad=False)
            indices = torch.arange(0, data.size(0), self.batch_size) 
            for i in range(len(indices)):
                start_idx = indices[i]
                end_idx = data.size(0) if i == len(indices) - 1 else indices[i+1]
                data_batch = data[start_idx:end_idx].to(self.device)
                result[start_idx:end_idx] = self.net.inference(data_batch).cpu()

            return result.clone().detach()
        
        def inverse_transform(self, data):
            data = data.to(self.device)
            return self.net.U(data)

    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device
        self.out_dim, self.hid_dim = args.out_dim, args.hid_dim
        self.train_batch_size = 64 # Define train batch size

        #MNIST data_matrix used for PCA
        train_data = torchvision.datasets.MNIST('./data/', train=True, download=True)
        train_data = train_data.data.to(torch.float32).reshape(len(train_data), -1)
        # train_data = torch.nn.functional.normalize(train_data, dim = 1)
        # to be consistent with the sparse training procedure
        train_data = train_data / 255
        train_data = (train_data - 0.1307) / 0.3081
        # plt.imshow(train_data[0].reshape([28,28]) + torch.randn([28,28])*0.1)
        # savefig(path="./image/MNIST", filename="_digit")

        # apply filter
        if self.args.filter != "none":
            if self.args.filter == "pca":
                logging.info("using PCA filter")
                self.ff_filter = PCA(n_components=self.out_dim) 
            elif self.args.filter == "sparse":
                logging.info("using sparse filter")
                self.ff_filter = self.SparseCoding(args.sparse_weight_path, self.device)
                self.out_dim = self.ff_filter.feature_dim
            else:
                raise NotImplementedError("Filter not implemented.")
        
            logging.info('begin fitting')
            self.hidden_states = self.ff_filter.fit_transform(train_data)
            logging.info('fitting complete')
        else:
            self.hidden_states = train_data
            self.out_dim = len(train_data[0])

        # self.std = np.sqrt(np.var(self.hidden_states, axis=0))
        # self.hidden_states = self.hidden_states/self.std
        # recovered_hid = self.pca.inverse_transform(self.hidden_states[0])
        # plt.imshow(recovered_hid.reshape([28,28]))
        # savefig(path="./image/MNIST", filename="_digit_recovered")
        # logging.info(self.hidden_states.shape)

    def train(self):
        train_dataset = torch.utils.data.TensorDataset(torch.tensor(self.hidden_states))
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.train_batch_size, shuffle=True)
        
        model = self.set_model()
        # model.apply(model.init_weights)
        # annealing noise
        n_level = 10
        noise_levels = [1/math.exp(math.log(1000)*n/n_level) for n in range(n_level)]

        nepoch = self.args.nepochs
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)
        # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

        # load weights
        if self.args.resume:
            load(f"./model/MNIST/{model.__class__.__name__}_MNIST_chkpt{self.args.run_id}", model, optimizer)
            model.set_weight()

        for epoch in tqdm(range(nepoch), dynamic_ncols=True):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, optimizer, f"./model/MNIST/{self.args.run_id}", f"{model.__class__.__name__}_MNIST_ep{epoch}")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.0001)

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
        with torch.no_grad():
            model = self.set_model()

            # load(f"./model/MNIST/{model.__class__.__name__}_MNIST_chkpt{self.args.run_id}", model)
            load(f"./model/MNIST/{self.args.run_id}/{model.__class__.__name__}_MNIST_ep{280}", model)

            model.set_weight()
            # samples = (torch.rand([10, self.hid_dim])-.5).to(self.device)
            if self.args.filter == "none":
                samples = torch.pinverse(model.W_out.weight)
                samples = (self.hidden_states[:10]@samples.T).to(self.device)
            elif self.args.model=="SO":
                samples = (torch.rand([10, self.out_dim])-.5).to(self.device)/1000
            elif self.args.model == "SR":
                samples = (torch.rand([10, self.hid_dim])-.5).to(self.device)/1000
            model.dt = 1e-6
            model = model.to(self.device)
            # samples = self.anneal_gen_sample(samples, 500)
            samples = gen_sample(model, samples, 10000)
            samples = model.W_out(samples)
            samples = samples.detach().cpu().numpy()
            # samples = samples*self.std
            if self.args.filter != "none":
                samples = self.ff_filter.inverse_transform(samples)
            samples = samples.reshape(len(samples), 28, 28)
            print(samples.shape)
            fig, axes = plt.subplots(2, 5)
            for i in range(2):
                for j in range(5):
                    ax = axes[i,j]
                    ax.imshow(samples[i*5+j])
            savefig(path="./image/MNIST", filename=self.args.model+"_digit_sampled")

    def anneal_gen_sample(self, initial_state, length):
        next = initial_state
        n_level = 10
        
        for i in range(length):
            if i % (length//n_level) ==0:
                model = self.set_model()
                load(f"./model/MNIST/{self.args.run_id}/{model.__class__.__name__}_MNIST_ep{(i//(length//n_level))*40}", model)
                model.set_weight()
            next = model(next)
        return next
    
    def set_model(self):
        if self.args.model == "SR":
            print("Using reservoir-sampler arch")
            model = rand_RNN(self.args.hid_dim, self.args.out_dim)
        elif self.args.model == "SO":
            print("Using sampler-only arch")
            model = FR(self.args.out_dim)
        else:
            return None

        return model.to(self.device)
