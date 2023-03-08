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
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
        
        # set up the model
        model = rand_RNN(self.args.hid_dim, train_dataset.out_dim).to(self.device)
        # model = torch.nn.DataParallel(model).to(self.args.device)

        # annealing noise
        n_level = 10
        noise_levels = [10/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

        # train the model
        nepoch = self.args.nepochs
        if self.args.resume:
            load(f"./model/{model.__class__.__name__}_celegans_ep{nepoch}", model)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        for epoch in tqdm(range(nepoch)):
            if epoch % (nepoch//n_level) ==0:
                noise_level = noise_levels[epoch//(nepoch//n_level)]
                logging.info(f"noise level: {noise_level}")
                save(model, f"./model/{self.args.run_id}", f"{model.__class__.__name__}_celegans_ep{epoch}")
                # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
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
        save(model, f"./model/{self.args.run_id}", f"{model.__class__.__name__}_celegans_ep{epoch}") 

    def test(self):
        # set up dataloader
        dataset = CelegansData()
        activity = dataset.activity_worms
        name_list = dataset.name_list

        # load model weights and set model
        model = rand_RNN(self.args.hid_dim, dataset.out_dim).to(self.args.device)
        load(f"./model/{self.args.run_id}/{model.__class__.__name__}_celegans_ep{self.args.nepochs}", model)
        with torch.no_grad():
            model.set_weight()
            initial_state = self.get_initial_state(model, activity)
            model.dt = 1e-3
            _, trace = self.gen_trace(model, initial_state, 774, dataset)
            trace = trace.detach().cpu().numpy()
            color_list = ['green','red']
            num_neuron = 6
            t_steps = 774
            time = np.arange(0,t_steps)
            for trial in range(21):
                logging.info(f"generating trial {trial}")
                _, axes = plt.subplots(num_neuron//2, 2, figsize=(25,10))
                for neuron_index in range(num_neuron):
                    ax = axes[neuron_index//2, neuron_index%2]
                    ax.plot(time,activity[:t_steps,trial,neuron_index],label='true',color=color_list[0])
                    ax.plot(time,trace[:t_steps,trial,neuron_index],label='generated',color=color_list[1])
                    data = np.zeros([2, t_steps])
                    data[0, :] = activity[:t_steps,trial,neuron_index]
                    data[1, :] = trace[:t_steps,trial,neuron_index]
                    corrcoef = np.corrcoef(data)
                    # logging.info(f"{corrcoef.shape}")
                    corrcoef = corrcoef[0,1]
                    neuron_name = name_list[neuron_index]
                    ax.set_title(f"neuron:{neuron_name}, corrcoef:{corrcoef}")
                    ax.legend()
                savefig(filename=f"celegans_trace_trial{trial}.png")
                plt.close()

    '''
        get the initial state for the hidden states
    '''
    @staticmethod
    def get_initial_state(model: rand_RNN, activity):
        init_out = torch.tensor(activity[0, :, :]).to(model.W_out.weight).T
        return torch.linalg.lstsq(model.W_out.weight, init_out).solution.T

    @staticmethod
    def gen_trace(model: rand_RNN, initial_state, length, dataset: CelegansData, annealing_step=100):
        odor = torch.tensor(dataset.odor_worms).to(initial_state)
        activity = dataset.activity_worms
        init_out = torch.tensor(activity[0, :, :]).to(model.W_out.weight)
        with torch.no_grad():
            ntrial = initial_state.shape[0]
            hidden_list = torch.zeros(length, ntrial, model.hid_dim)
            trace = torch.zeros(length, ntrial, model.out_dim)
            hidden_list[0] = initial_state
            trace[0] = init_out
            next = initial_state
            for i in range(1, length*annealing_step):
                idx = i//annealing_step
                next = model(next, odor[idx, :, :])
                if i%annealing_step==0:
                    hidden_list[idx] = next
                    trace[idx] = model.W_out(next) + model.true_input(odor[idx, :, :])
            return hidden_list, trace
            

