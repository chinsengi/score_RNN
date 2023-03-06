import torch
from utility import *
from models import rand_RNN
from torch.utils.data import Dataset
import random
import shutil
import logging
import tensorboardX
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from os.path import dirname, join as pjoin

__all__ = ['Celegans']

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
        #trace_arr = trace_variable['norm_traces']
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
        activity_worms = trace_datasets[:,:, T_start:774]

        step = 0.25
        time = np.arange(start = 0, stop = T * step , step = step)
        # odor list
        odor_list = ['butanone','pentanedione','NaCL']
        # multiple odor datasets concatnate
        for idata in range(N_dataset):
            for it_stimu in range(stimulate_seconds.shape[0]):
                tim1_ind = time>stimulate_seconds[it_stimu][0]
                tim2_ind = time<stimulate_seconds[it_stimu][1]
                odor_on = np.multiply(tim1_ind.astype('int'),tim2_ind.astype('int'))
                stim_odor = stims[idata][it_stimu] - 1
                odor_datasets[idata][stim_odor][:] = odor_on
                        
        odor_worms = odor_datasets[:,:, T_start:774]

        self.activity_worms = np.swapaxes(activity_worms, 1, 2).reshape(-1, 189)
        self.odor_worms = np.swapaxes(odor_worms, 1, 2).reshape(-1, 3)
        self.in_dim = self.odor_worms.shape[1]
        self.out_dim = self.activity_worms.shape[1]

    def __len__(self):
        return self.odor_worms.shape[0]
    
    def __getitem__(self, idx):
        return self.odor_worms[idx], self.activity_worms[idx]

    

class Celegans():
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device

    def train(self):
        # set up tensorboard logging
        tb_path = os.path.join(self.args.run, 'tensorboard', self.args.run_id)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # set up dataloader
        train_dataset = CelegansData()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64)
        
        # set up the model
        model = rand_RNN(self.args.hid_dim, train_dataset.out_dim).to(self.args.device)
        # model = torch.nn.DataParallel(model).to(self.args.device)

        # annealing noise
        n_level = 10
        noise_levels = [10/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        # train the model
        nepoch = 400
        if self.args.resume:
            load(f"./model/{model.__class__.__name__}_celegans_ep{nepoch}", model)
        model.train()
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_celegans_ep{epoch+1}")
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
            for step, (odor, h) in enumerate(train_loader):
                h = h.to(self.args.device, torch.float32)
                odor = odor.to(self.args.device, torch.float32)
                h_noisy = h + torch.randn_like(h)*noise_level
                loss = 0.5*((model.score(h_noisy, odor) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

                #backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar('loss', loss, global_step=step)
                # logging.info("step: {}, loss: {}".format(step, loss.item()))

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_celegans_ep{nepoch+1}")   

    def test(self):
        pass
    