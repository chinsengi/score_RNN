import torch
from torch.utils.data import Dataset
import random
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
from torch.distributions import MultivariateNormal, MixtureSameFamily

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
    def __init__(self, GMMmean, GMMstd, n=1000):
        super().__init__()
        self.n_comp = GMMmean.shape[0]
        self.n_feature = GMMmean.shape[1]
        self.GMMmean = GMMmean
        self.GMMstd = GMMstd
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mode_id = random.randint(0, self.n_comp-1)
        GMMdata =  torch.normal(self.GMMmean[mode_id, :], self.GMMstd[mode_id, :])
        return GMMdata

class CelegansData(Dataset):
    def __init__(self):
        # load datasets
        N_dataset = 21
        N_cell = 189
        T = 960
        N_length = 109
        odor_channels = 3
        T_start = 160
        trace_datasets = np.zeros((N_dataset, N_cell, T))
        odor_datasets = np.zeros((N_dataset, odor_channels, T))
                
        # .mat data load
        basepath = 'data/worm_activity'
        mat_fname = pjoin(basepath, 'all_traces_Heads_new.mat')
        trace_variable = sio.loadmat(mat_fname)
        is_L = trace_variable['is_L']
        neurons_name = trace_variable['neurons']
        trace_arr = trace_variable['traces']
        stimulate_seconds = trace_variable['stim_times']
        stims = trace_variable['stims']
        # multiple trace datasets concatnate
        for idata in range(N_dataset):
            ineuron = 0
            for ifile in range(N_length):
                if trace_arr[ifile][0].shape[1] == 42:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                    data = trace_arr[ifile][0][0][idata + 21]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
                else:
                    data = trace_arr[ifile][0][0][idata]
                    if data.shape[0] < 1:
                        trace_datasets[idata][ineuron][:] = np.nan
                    else:
                        trace_datasets[idata][ineuron][0:data[0].shape[0]] = data[0]
                    ineuron+= 1
        # neural activity target
        activity_worms = trace_datasets[:,:, T_start:]
        name_list = []
        for ifile in range(N_length):
            if is_L[ifile][0][0].shape[0] == 42:
                name_list.append(neurons_name[ifile][0][0] + 'L')
                name_list.append(neurons_name[ifile][0][0] + 'R')
            else:
                name_list.append(neurons_name[ifile][0][0])
        self.name_list = name_list

        # odor list
        step = 0.25
        time = np.arange(start = 0, stop = T * step , step = step)
        odor_list = ['butanone','pentanedione','NaCL']
        # multiple odor datasets concatnate
        for idata in range(N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][0]
                tim2_ind = time<stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype('int'),tim2_ind.astype('int'))
                stim_odor = stims[idata][it_stimu] - 1
                odor_datasets[idata][stim_odor][:] = odor_on
                        
        odor_worms = odor_datasets[:,:, T_start:]

        # set internal data
        self.activity_worms = np.moveaxis(activity_worms[:,:,:774], [0,1,2], [1,2,0]) # [time, trial, neuron]
        self.odor_worms = np.moveaxis(odor_worms[:,:,:774], [0,1,2], [1,2,0]) # [time, trial, odor]
        self.activity = self.activity_worms.reshape(-1, 189)
        self.odor = self.odor_worms.reshape(-1, 3)
        self.in_dim = self.odor.shape[1]
        self.out_dim = self.activity.shape[1]

    def __len__(self):
        return self.activity.shape[0]
    
    def __getitem__(self, idx):
        return self.odor[idx], self.activity[idx]
