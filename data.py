import torch
from torch.utils.data import Dataset
import random

# generate initial states that conforms to the Gaussian distribution
class GaussianData(Dataset):
    def __init__(self, mean = 0, variance = 1, n=1000, ndim=2):
        super().__init__()
        self.mean = mean
        self.variance = variance
        self.n = n
        self.ndim = ndim   

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        return torch.empty(self.ndim).normal_(mean=self.mean,std=self.variance)

class UniformData(Dataset):
    def __init__(self, lower, upper, n=1000, ndim=2):
        super().__init__()
        self.ndim = ndim
        self.lower = lower
        self.upper = upper
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        return torch.rand(self.ndim)*(self.upper-self.lower) + self.lower

class GMMData(Dataset):
    def __init__(self, GMMmean, GMMvar, mean, var, n=1000):
        super().__init__()
        self.n_comp = len(GMMmean)
        self.GMMmean = GMMmean
        self.GMMvar = GMMvar
        self.mean = mean
        self.var = var
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mode_id = random.randint(0, self.n_comp-1)
        GMMdata =  torch.normal(torch.cat([self.GMMmean[:,mode_id], self.mean]), \
            torch.cat([self.GMMvar[:,mode_id], self.var]))
        GMMdata = GMMdata.reshape(GMMdata.shape[0],-1)
        return GMMdata