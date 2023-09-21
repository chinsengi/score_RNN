import torch
from utility import *
from models import CelegansRNN
from torch.utils.data import Dataset
import shutil
import logging
from scipy.stats import entropy, wasserstein_distance
import seaborn as sns
import pandas as pd
import torch.nn as nn

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
        # get connectome data
        self.connectome = WhiteConnectomeData("./data/worm_cnct", device=self.device)

        # set up dataloader
        self.dataset = CelegansData(self.connectome, self.device)

        self.non_lin = set_nonlin(args.nonlin)

    def train(self):
        # set up tensorboard logging
        tb_path = os.path.join(self.args.run, "tensorboard", self.args.run_id)
        if os.path.exists(tb_path):
            shutil.rmtree(tb_path)

        tb_logger = tensorboardX.SummaryWriter(log_dir=tb_path)

        # get measured neuron info
        observed_mask = self.dataset.observed_mask

        # set up the model
        model = CelegansRNN(
            self.connectome, self.dataset.odor_dim, non_lin=self.non_lin
        ).to(self.device)
        # model = CelegansRNN(self.connectome, self.dataset.odor_dim).to(self.device)

        # set up dataloader
        train_loader = torch.utils.data.DataLoader(
            self.dataset, batch_size=64, shuffle=True
        )

        # annealing noise
        n_level = 10
        noise_levels = [
            2 / math.exp(math.log(100) * (n + 1) / n_level) for n in range(n_level)
        ]

        # train the model
        nepoch = self.args.nepochs
        model.train()
        # model = torch.compile(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        if self.args.resume:
            load(
                f"./model/Celegans/{self.args.run_id}/{model.__class__.__name__}_chkpt.pth",
                model,
                optimizer,
            )
        for epoch in tqdm(range(nepoch)):
            # decrease the noise level
            if epoch % (nepoch // n_level) == 0:
                noise_level = noise_levels[epoch // (nepoch // n_level)]
                logging.info(f"noise level: {noise_level}")
                save(
                    model,
                    optimizer,
                    f"./model/Celegans/{self.args.run_id}",
                    f"{model.__class__.__name__}_celegans_ep{epoch}",
                )

            # train the model based on fake y distribution
            for step, (odor, h) in enumerate(train_loader):
                h = h.to(self.args.device, torch.float32)
                odor = odor.to(self.args.device, torch.float32)
                h_noisy = h + torch.randn_like(h) * noise_level
                loss = 0.5 * (
                    (model.score(h_noisy, odor) - (h - h_noisy) / noise_level**2) ** 2
                )[:, observed_mask].sum(dim=-1).mean(dim=0)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                tb_logger.add_scalar("loss", loss, global_step=step)
                # logging.info("step: {}, loss: {}".format(step, loss.item()))
            if (
                epoch + 1
            ) % self.args.impute_freq == 0 and not self.args.disable_impute:
                self.dataset.reimpute(model, 80)
            logging.info(f"loss: {loss.item():>7f}, Epoch: {epoch}")

        save(
            model,
            optimizer,
            f"./model/Celegans/{self.args.run_id}",
            f"{model.__class__.__name__}_chkpt.pth",
        )

    def test(self):
        # retrieve the data
        activity = self.dataset.activity_worms
        name_list = self.dataset.name_list

        # load model weights and set model
        model = CelegansRNN(
            self.connectome, self.dataset.odor_dim, non_lin=self.non_lin
        ).to(self.device)
        load(
            f"./model/Celegans/{self.args.run_id}/{model.__class__.__name__}_chkpt.pth",
            model,
        )
        with torch.no_grad():
            warmup = 0
            timestep=10000
            trace = self.dataset.reconstruct(
                model, warmup=warmup, timestep=timestep, dt=1e-4
            )[:, :, self.dataset.observed_mask]
            trace = trace.detach().cpu().numpy()
            n_observed = np.sum(self.dataset.observed_mask).item()
            w_dist = np.zeros((2, n_observed))
            for i in range(n_observed):
                w_dist[0, i] = wasserstein_distance(
                    trace[:80, :, i].flatten(), activity[:80, :, i].flatten()
                )
                w_dist[1, i] = wasserstein_distance(
                    activity[:80, :11, i].flatten(), activity[:80, 11:, i].flatten()
                )
            df = pd.DataFrame(w_dist.T, columns=["recovered", "baseline"])
            df.plot(kind="box")
            savefig(
                path="./image/celegans",
                filename=f"Wasserstein_distribution{self.args.run_id}",
            )
            # plot the histogram to visualize and compare the distribution
            # print(f'the max: {np.max(trace[:60,:,:])}')
            # breakpoint()
            num_neuron = 16
            t_steps = 80
            ncol = 4
            _, axes = plt.subplots(num_neuron // ncol, ncol, figsize=(25, 25))
            for neuron_index in range(num_neuron):
                ax = axes[neuron_index // ncol, neuron_index % ncol]
                true_data_1 = activity[:t_steps, :11, neuron_index].flatten()
                true_data_2 = activity[:t_steps, 11:, neuron_index].flatten()
                gen_data = trace[(timestep-80):timestep, :, neuron_index].flatten()
                ax.hist(
                    true_data_1,
                    density=True,
                    alpha=0.5,
                    bins=np.linspace(min(true_data_1), max(true_data_1), 20),
                )
                ax.hist(
                    gen_data,
                    density=True,
                    alpha=0.5,
                    bins=np.linspace(min(true_data_1), max(true_data_1), 20),
                )
                ax.hist(
                    true_data_2,
                    density=True,
                    alpha=0.5,
                    bins=np.linspace(min(true_data_1), max(true_data_1), 20),
                )
                ax.legend(["true_1-11", "generated", "true_trial12-21"])
            savefig(
                path="./image/celegans",
                filename=f"celegans_trace_before1st_{self.args.run_id}",
            )

            # plot the reconstruction after the first odor
            # t_steps = 200
            # start = 361
            # trace = self.dataset.reconstruct(
            #     model, start=start, warmup=warmup, timestep=t_steps, dt=1e-4
            # )[:, :, self.dataset.observed_mask]
            # trace = trace.detach().cpu().numpy()
            # num_neuron = 16
            # ncol = 4
            # _, axes = plt.subplots(num_neuron // ncol, ncol, figsize=(25, 25))
            # for neuron_index in range(num_neuron):
            #     ax = axes[neuron_index // ncol, neuron_index % ncol]
            #     true_data_1 = activity[
            #         start + t_steps - 40 : start + t_steps, :, neuron_index
            #     ].flatten()
            #     gen_data = trace[
            #         t_steps - 40 :t_steps, :, neuron_index
            #     ].flatten()
            #     ax.hist(
            #         true_data_1,
            #         density=True,
            #         alpha=0.5,
            #         bins=np.linspace(min(true_data_1), max(true_data_1), 20),
            #     )
            #     ax.hist(
            #         gen_data,
            #         density=True,
            #         alpha=0.5,
            #         bins=np.linspace(min(true_data_1), max(true_data_1), 20),
            #     )
            #     ax.legend(["true", "generated"])
            # savefig(
            #     path="./image/celegans",
            #     filename=f"celegans_trace_after1st_{self.args.run_id}",
            # )

            # n_observed = np.sum(self.dataset.observed_mask).item()
            # w_dist = np.zeros((2 * (t_steps // 40) + 1, n_observed))
            # for i in range(n_observed):
            #     w_dist[0, i] = wasserstein_distance(
            #         activity[start : start + t_steps, :11, i].flatten(),
            #         activity[start : start + t_steps, 11:, i].flatten(),
            #     )
            #     for j in range(t_steps // 40):
            #         w_dist[2 * j + 1, i] = wasserstein_distance(
            #             activity[
            #                 start + 40 * j : start + 40 * (j + 1), :11, i
            #             ].flatten(),
            #             activity[
            #                 start + 40 * j : start + 40 * (j + 1), 11:, i
            #             ].flatten(),
            #         )
            #         w_dist[2 * j + 2, i] = wasserstein_distance(
            #             trace[40 * j : 40 * (j + 1), :, i].flatten(),
            #             activity[start + 40 * j : start + 40 * (j + 1), :, i].flatten(),
            #         )
            #     breakpoint()
            # df = pd.DataFrame(
            #     w_dist.T,
            #     # columns=["baseline"]
            #     # + [
            #     #     f"{start + 40 * j} - {start + 40 * (j + 1)}"
            #     #     for j in range(t_steps // 40)
            #     # ],
            # )
            # df.plot(kind="box", showfliers=False)
            # savefig(
            #     path="./image/celegans",
            #     filename=f"Wasserstein_distribution_after1stodor{self.args.run_id}",
            # )
            # for trial in range(21):
            #     logging.info(f"generating trial {trial}")
            #     _, axes = plt.subplots(num_neuron // 2, 2, figsize=(25, 10))
            #     for neuron_index in range(num_neuron):
            #         ax = axes[neuron_index // 2, neuron_index % 2]
            #         ax.plot(
            #             time,
            #             activity[:t_steps, trial, neuron_index],
            #             label="true",
            #             color=color_list[0],
            #         )
            #         ax.plot(
            #             time,
            #             trace[:t_steps, trial, neuron_index],
            #             label="generated",
            #             color=color_list[1],
            #         )
            #         data = np.zeros([2, t_steps])
            #         data[0, :] = activity[:t_steps, trial, neuron_index]
            #         data[1, :] = trace[:t_steps, trial, neuron_index]
            #         corrcoef = np.corrcoef(data)
            #         # logging.info(f"{corrcoef.shape}")
            #         corrcoef = corrcoef[0, 1]
            #         neuron_name = name_list[neuron_index]
            #         ax.set_title(f"neuron:{neuron_name}, corrcoef:{corrcoef}")
            #         ax.legend()
            #     savefig(path='./image/celegans', filename=f"celegans_trace_trial{trial}")
            #     plt.close()

    """
        get the distribution of a vector of sample
    """

    @staticmethod
    def get_batch_distribution(samples, bin_edges):
        n_neuron = samples.shape[-1]
        distributions = np.zeros((n_neuron, len(bin_edges) - 1))
        # Calculate the histogram with probabilities (density=True)
        for i in range(n_neuron):
            distributions[i], _ = np.histogram(
                samples[..., i], bins=bin_edges, density=True
            )

        return distributions * (bin_edges[1] - bin_edges[0])
