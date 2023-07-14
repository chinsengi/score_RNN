import torch
from utility import *
from models import CelegansRNN
from torch.utils.data import Dataset
import shutil
import logging

try:
    import tensorboardX
except ModuleNotFoundError:
    pass
import numpy as np
import matplotlib.pyplot as plt
from data import CelegansData
from connectome_preprocess import WhiteConnectomeData

__all__ = ["Celegans"]


class Celegans:
    def __init__(self, args) -> None:
        self.args = args
        self.device = args.device

    def train(self):
        # set up tensorboard logging
        tb_path = os.path.join(self.args.run, "tensorboard", self.args.run_id)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # get connectome data
        connectome = WhiteConnectomeData("./data/worm_cnct", device=self.device)

        # set up dataloader
        train_dataset = CelegansData()
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=64, shuffle=True
        )

        # get sensory neuron info
        n_observed = train_dataset.activity_worms.shape[-1]
        observed_mask = torch.tensor(
            [
                1 if n in train_dataset.name_list else 0
                for n in connectome.neuron_names
            ]
        )

        # set up the model
        model = CelegansRNN(
            n_observed,
            observed_mask,
            connectome,
        ).to(self.device)
        # model = torch.nn.DataParallel(model).to(self.args.device)

        # annealing noise
        n_level = 10
        noise_levels = [
            5 / math.exp(math.log(100) * n / n_level) for n in range(n_level)
        ]

        # train the model
        nepoch = self.args.nepochs
        model.train()
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        if self.args.resume:
            load(f"./model/{model.__class__.__name__}_celegans_chkpt", model, optimizer)
        # initialize unobserved neurons with random values
        y = torch.randn(train_dataset.out_dim).to(self.args.device, torch.float32)
        for epoch in tqdm(range(nepoch)):
            # decrease the noise level
            if epoch % (nepoch // n_level) == 0:
                noise_level = noise_levels[epoch // (nepoch // n_level)]
                logging.info(f"noise level: {noise_level}")
                save(
                    model,
                    optimizer,
                    f"./model/{self.args.run_id}",
                    f"{model.__class__.__name__}_celegans_ep{epoch}",
                )

            # train the model based on fake y distribution
            for step, (odor, h) in enumerate(train_loader):
                h = h.to(self.args.device, torch.float32)
                odor = odor.to(self.args.device, torch.float32)
                h_noisy = h + torch.randn_like(h) * noise_level
                loss = 0.5 * (
                    (model.score(h_noisy, odor) - (h - h_noisy) / noise_level**2) ** 2
                ).sum(dim=-1).mean(dim=0)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                tb_logger.add_scalar("loss", loss, global_step=step)
                # logging.info("step: {}, loss: {}".format(step, loss.item()))

            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")
        save(
            model,
            optimizer,
            f"./model/{self.args.run_id}",
            f"{model.__class__.__name__}_celegans_ep{epoch}",
        )

    def test(self):
        # set up dataloader
        dataset = CelegansData()
        activity = dataset.activity_worms
        name_list = dataset.name_list

        # load model weights and set model
        model = rand_RNN(self.args.hid_dim, dataset.out_dim).to(self.args.device)
        load(
            f"./model/{self.args.run_id}/{model.__class__.__name__}_celegans_chkpt",
            model,
        )
        with torch.no_grad():
            model.set_weight()
            initial_state = self.get_initial_state(model, activity)
            model.dt = 1e-3
            _, trace = self.gen_trace(model, initial_state, 774, dataset)
            trace = trace.detach().cpu().numpy()
            color_list = ["green", "red"]
            num_neuron = 6
            t_steps = 774
            time = np.arange(0, t_steps)
            for trial in range(21):
                logging.info(f"generating trial {trial}")
                _, axes = plt.subplots(num_neuron // 2, 2, figsize=(25, 10))
                for neuron_index in range(num_neuron):
                    ax = axes[neuron_index // 2, neuron_index % 2]
                    ax.plot(
                        time,
                        activity[:t_steps, trial, neuron_index],
                        label="true",
                        color=color_list[0],
                    )
                    ax.plot(
                        time,
                        trace[:t_steps, trial, neuron_index],
                        label="generated",
                        color=color_list[1],
                    )
                    data = np.zeros([2, t_steps])
                    data[0, :] = activity[:t_steps, trial, neuron_index]
                    data[1, :] = trace[:t_steps, trial, neuron_index]
                    corrcoef = np.corrcoef(data)
                    # logging.info(f"{corrcoef.shape}")
                    corrcoef = corrcoef[0, 1]
                    neuron_name = name_list[neuron_index]
                    ax.set_title(f"neuron:{neuron_name}, corrcoef:{corrcoef}")
                    ax.legend()
                savefig(filename=f"celegans_trace_trial{trial}.png")
                plt.close()

    """
        get the initial state for the hidden states
    """

    @staticmethod
    def get_initial_state(model: CelegansRNN, activity):
        init_out = torch.tensor(activity[0, :, :]).to(model.W_out.weight).T
        return torch.linalg.lstsq(model.W_out.weight, init_out).solution.T

    @staticmethod
    def gen_trace(
        model: CelegansRNN,
        initial_state,
        length,
        dataset: CelegansData,
        annealing_step=1,
    ):
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
            for i in range(1, length * annealing_step):
                idx = i // annealing_step
                next = model(next, odor[idx, :, :])
                if i % annealing_step == 0:
                    hidden_list[idx] = next
                    trace[idx] = model.W_out(next) + model.true_input(odor[idx, :, :])
            return hidden_list, trace
