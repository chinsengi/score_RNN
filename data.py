import torch
from torch.utils.data import Dataset
import random
from os.path import join as pjoin
import numpy as np
import scipy.io as sio
from torch.distributions import MultivariateNormal, MixtureSameFamily
from torch.distributions.laplace import Laplace
from mv_laplace import MvLaplaceSampler
from connectome_preprocess import WhiteConnectomeData


# generate initial states that conforms to the Gaussian distribution
class GaussianData(Dataset):
    def __init__(self, mean=0, variance=1, n=1000, ndim=2):
        super().__init__()
        self.mean = mean
        self.variance = variance
        self.n = n
        self.ndim = ndim

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, index):
        return torch.empty(self.ndim).normal_(mean=self.mean, std=self.variance)


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
        return torch.rand(self.ndim) * (self.upper - self.lower) + self.lower


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
        mode_id = random.randint(0, self.n_comp - 1)
        GMMdata = torch.normal(self.GMMmean[mode_id, :], self.GMMstd[mode_id, :])
        return GMMdata


class LAPData(Dataset):
    # LAPMean: 2d array, each row is a mean vector
    # LAPstd: 3d array, each element is a 2D PSD matrix
    def __init__(self, LAPmean, LAPstd, n=1000):
        super().__init__()
        self.n_comp = LAPmean.shape[0]
        self.n_feature = LAPmean.shape[1]
        self.LAPmean = LAPmean
        self.LAPstd = LAPstd
        self.n = n
        # create samplers
        self.samplers = []
        for i in range(self.n_comp):
            self.samplers.append(
                MvLaplaceSampler(self.LAPmean[i, :], self.LAPstd[i, :])
            )

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        mode_id = random.randint(0, self.n_comp - 1)
        return self.samplers[mode_id].sample(1)


class CelegansData(Dataset):
    def __init__(self, connectome: WhiteConnectomeData, device):
        super().__init__()
        self.device = device
        # load datasets
        N_dataset = 21
        N_cell = 189
        T = 960
        N_length = 109
        odor_channels = 3
        T_start = 160
        trace_datasets = np.zeros((N_dataset, N_cell, T))
        odor_datasets = np.zeros((N_dataset, odor_channels, T))
        self.total_neuron_cnt = connectome.num_neurons

        # .mat data load
        basepath = "data/worm_activity"
        mat_fname = pjoin(basepath, "all_traces_Heads_new.mat")
        trace_variable = sio.loadmat(mat_fname)
        is_L = trace_variable["is_L"]
        neurons_name = trace_variable["neurons"]
        trace_arr = trace_variable["traces"]
        stimulate_seconds = trace_variable["stim_times"]
        stims = trace_variable["stims"]
        # multiple trace datasets concatnate
        for data_idx in range(N_dataset):
            neuro_idx = 0
            for ifile in range(N_length):
                if trace_arr[ifile][0].shape[1] == 42:
                    data = trace_arr[ifile][0][0][data_idx]
                    if data.shape[0] < 1:
                        trace_datasets[data_idx][neuro_idx][:] = np.nan
                    else:
                        trace_datasets[data_idx][neuro_idx][
                            0 : data[0].shape[0]
                        ] = data[0]
                    neuro_idx += 1
                    data = trace_arr[ifile][0][0][data_idx + 21]
                    if data.shape[0] < 1:
                        trace_datasets[data_idx][neuro_idx][:] = np.nan
                    else:
                        trace_datasets[data_idx][neuro_idx][
                            0 : data[0].shape[0]
                        ] = data[0]
                else:
                    data = trace_arr[ifile][0][0][data_idx]
                    if data.shape[0] < 1:
                        trace_datasets[data_idx][neuro_idx][:] = np.nan
                    else:
                        trace_datasets[data_idx][neuro_idx][
                            0 : data[0].shape[0]
                        ] = data[0]
                neuro_idx += 1
        # neural activity target
        activity_worms = trace_datasets[:, :, T_start:]
        name_list = []
        for ifile in range(N_length):
            if is_L[ifile][0][0].shape[0] == 42:
                name_list.append(neurons_name[ifile][0][0] + "L")
                name_list.append(neurons_name[ifile][0][0] + "R")
            else:
                name_list.append(neurons_name[ifile][0][0])
        self.name_list = name_list

        # odor list
        step = 0.25
        time = np.arange(start=0, stop=T * step, step=step)
        odor_list = ["butanone", "pentanedione", "NaCL"]
        # multiple odor datasets concatnate
        for data_idx in range(N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time > stimulate_seconds[it_stimu][0]
                tim2_ind = time < stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype("int"), tim2_ind.astype("int"))
                stim_odor = stims[data_idx][it_stimu] - 1
                odor_datasets[data_idx][stim_odor][:] = odor_on

        odor_worms = odor_datasets[:, :, T_start:]

        # set internal data
        self.activity_worms = torch.tensor(
            np.moveaxis(activity_worms[:, :, :774], [0, 1, 2], [1, 2, 0])
        ).float()  # [time, trial, neuron]
        self.odor_worms = torch.tensor(
            np.moveaxis(odor_worms[:, :, :774], [0, 1, 2], [1, 2, 0])
        ).float()  # [time, trial, odor]
        self.observed_activity = self.activity_worms.reshape(-1, 189)
        self.odor = self.odor_worms.reshape(-1, 3)
        self.odor_dim = self.odor.shape[1]

        self.observed_mask = [(n in self.name_list) for n in connectome.neuron_names]

        # initialize the missing activity with gaussian noise using the mean and variance of the observed activity
        # mean_activity = self.observed_activity.mean()
        # std_activity = self.observed_activity.std()
        # self.all_activity = (
        #     torch.randn(
        #         (
        #             self.activity_worms.shape[0],
        #             self.activity_worms.shape[1],
        #             connectome.num_neurons,
        #         )
        #     )
        #     * std_activity
        #     + mean_activity
        # )

        # initialize the missing activity with 0 and observed activity with the true value
        self.all_activity = torch.zeros(
            (
                self.activity_worms.shape[0],
                self.activity_worms.shape[1],
                connectome.num_neurons,
            )
        )
        self.all_activity[:, :, self.observed_mask] = self.activity_worms
        self.activity_samples = self.all_activity.reshape(-1, connectome.num_neurons)

    def reimpute(self, model):
        n_trials = self.activity_worms.shape[1]
        n_timestep = self.activity_worms.shape[0]
        for trial in range(n_trials):
            for t in range(n_timestep - 1):
                self.all_activity[t + 1, trial, :] = model(
                    self.all_activity[t, trial, :].unsqueeze(0).to(self.device),
                    self.odor_worms[t, trial, :].unsqueeze(0).to(self.device),
                ).squeeze(0).detach()
                self.all_activity[
                    t + 1, trial, self.observed_mask
                ] = self.activity_worms[t + 1, trial, :]
        self.activity_samples = self.all_activity.reshape(-1, self.total_neuron_cnt)

    def __len__(self):
        return self.activity_samples.shape[0]

    def __getitem__(self, idx):
        return self.odor[idx], self.activity_samples[idx]
