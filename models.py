import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math
from torch.autograd.functional import jacobian
from utility import *
import torch.nn.init as init


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=32, nlayer=2):
        super(MLP, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.nlayer = nlayer
        self.w_in = nn.Linear(in_dim, hid_dim)
        self.w_out = nn.Linear(hid_dim, out_dim)
        self.w_hh = []
        for i in range(self.nlayer):
            self.w_hh.append(nn.Linear(hid_dim, hid_dim))

    def forward(self, input):
        input = torch.flatten(input)
        x = self.w_in(input)
        for i in range(self.nlayer):
            x = torch.tanh(x)
            x = self.w_hh[i](x)
        x = torch.tanh(x)
        return self.w_out(x)


"""
vanilla synaptic current model of RNN, without output layer
"""


class SynCurrentDyn(torch.nn.Module):
    def __init__(self, hid_dim, dt=0.001):
        super().__init__()
        self.hid_dim = hid_dim
        self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Identity()
        self.non_lin = nn.ReLU()
        self.dt = dt
        self.is_set_weight = True

    def forward(self, input):
        v = self.score(input)
        # mean = torch.zeros(self.hid_dim-1).to(input)-2
        # var  = torch.ones(self.hid_dim-1).to(input)/2
        # GMM_mean = torch.tensor([-2, 2 ]).to(input).unsqueeze(0)
        # GMM_var = torch.tensor([0.5, .5]).to(input).unsqueeze(0)
        # v = torch.cat((score_GMM(input[:,[0]], GMM_mean, GMM_var), score_normal(input[:,1:], mean, var)), 1)
        # v = score_normal(input,mean, var)
        return input + self.dt * v + math.sqrt(2 * self.dt) * torch.randn_like(input)

    def score(self, input):
        input_trans = self.non_lin(input)
        v = self.W(input_trans)
        # v = v - torch.diag(self.W.weight)*input_trans
        # v = self.W(input)
        # v = self.W(torch.tanh(v))
        # v = v - torch.diag(self.W2.weight)*torch.tanh(v)
        v = v - self.gamma.T * input
        return v

    def set_weight(self):
        pass

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)


"""
vanilla synaptic current & firing rate model of RNN, without output layer
"""


class NeuralDyn(torch.nn.Module):
    def __init__(self, hid_dim, synap=True, dt=0.001, non_lin=nn.ReLU()):
        super().__init__()
        self.hid_dim = hid_dim
        self.gamma = Parameter(torch.ones(self.hid_dim, 1, requires_grad=True))
        self.sig = Parameter(
            torch.eye(self.hid_dim, requires_grad=True)
        )  # actually inverse diffusion coefficient
        self.W = nn.Linear(self.hid_dim, self.hid_dim, bias=True)
        self.W_out = nn.Identity()
        self.non_lin = non_lin
        self.dt = dt

        # if the dynamics is synaptic current or not (firing rate dynamics)
        self.synap = synap
        self.is_set_weight = True

    def forward(self, input):
        v = self.score(input)
        return input + self.dt * v + math.sqrt(2 * self.dt) * torch.randn_like(input)

    def score(self, input):
        if self.synap:
            input_trans = self.non_lin(input)
            v = self.W(input_trans)
        else:
            input_trans = self.W(input)
            v = self.non_lin(input_trans)
        v = v - input
        v = self.gamma.T * v
        v = v @ self.sig @ self.sig.T
        return v

    def set_weight(self):
        pass

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)


class rand_RNN(torch.nn.Module):
    def __init__(
        self, hid_dim, out_dim, dt=0.001, non_lin=nn.ReLU(), fast_sampling=False
    ):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        # self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Linear(hid_dim, out_dim, bias=False)
        self.W1 = nn.Linear(hid_dim, out_dim, bias=False)
        self.W2 = nn.Linear(out_dim, hid_dim, bias=True)
        self.J = Parameter(torch.randn(out_dim, out_dim, requires_grad=True))
        self.is_set_weight = False
        self.non_lin = non_lin
        # self.non_lin = nn.LeakyReLU(0.1)
        # self.non_lin = torch.nn.Tanh()
        self.dt = dt
        self.fast_sampling = fast_sampling

    def forward(self, hidden):
        v = self.cal_v(hidden)
        nbatch = hidden.shape[0]
        return (
            hidden
            + self.dt * v
            + (math.sqrt(2 * self.dt) * torch.randn(nbatch, self.out_dim).to(hidden))
            @ self.sig.T
        )

    def set_weight(self):
        W_rec_tilde = self.W2.weight
        self.W_out.weight = self.W1.weight
        self.sig = torch.linalg.solve(
            self.W1.weight @ self.W1.weight.T, self.W1.weight.T, left=False
        )
        self.W_rec.weight = Parameter(W_rec_tilde @ self.W1.weight)
        self.W_rec.bias = Parameter(self.W2.bias)
        self.is_set_weight = True

    # calculate the dynamics of the hidden state
    def cal_v(self, hidden):
        v = -hidden + self.non_lin(self.W_rec(hidden))
        return v

    # calculate the dynamics (score function) of the output
    def score(self, sample):
        internal_score = -sample + self.W1(self.non_lin(self.W2(sample)))
        if self.fast_sampling:
            skew_symmetric = self.get_skew_symmetric()
            # lam = skew_symmetric[0,1].detach()
            # internal_score = internal_score @ (torch.eye(self.out_dim).to(sample) - skew_symmetric)/(1+lam**2)
            internal_score = internal_score @ torch.linalg.solve(
                torch.eye(self.out_dim).to(sample) - skew_symmetric
            )
        return internal_score

    def get_skew_symmetric(self):
        skew_symmetric = self.J - self.J.T
        skew_symmetric = 0.01 * skew_symmetric / torch.norm(skew_symmetric)
        return skew_symmetric

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)


class CelegansRNN(torch.nn.Module):
    def __init__(self, connectome, sensory_input_dim, dt=0.001, non_lin=nn.Softplus()):
        super().__init__()
        self.hid_dim = connectome.num_neurons  # number of all neurons
        # self.observed_mask = observed_mask  # mask for measured (observed) neurons
        self.gamma = Parameter(torch.ones(1, self.hid_dim, requires_grad=True))
        self.v_rest = Parameter(torch.zeros(1, self.hid_dim, requires_grad=True))
        self.W_elec = Parameter(torch.zeros(self.hid_dim, self.hid_dim))
        self.sig = Parameter(torch.eye(self.hid_dim, requires_grad=True))
        self.W_chem = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.E = nn.Linear(self.hid_dim, self.hid_dim, bias=False)
        self.J = Parameter(torch.zeros(self.hid_dim, self.hid_dim, requires_grad=True))
        self.connectome = connectome
        self.non_lin = non_lin
        # self.non_lin = nn.LeakyReLU(0.1)
        # self.non_lin = nn.Tanh()
        self.dt = dt

        # sensory mask
        self.sensory_mask = connectome.neuron_mask_dict["sensory"].bool().squeeze()
        self.in_dim = int(
            torch.sum(self.sensory_mask).item()
        )  # number of sensory neurons

        # sensory input
        self.Win = nn.Sequential(
            nn.Linear(sensory_input_dim, self.hid_dim * 2),
            non_lin,
            nn.Linear(self.hid_dim * 2, self.in_dim),
        )
        self.init_weights()

    def forward(self, hidden, input=None):
        F = self.cal_F(hidden, input)
        nbatch = hidden.shape[0]
        return (
            hidden
            + self.dt * F
            + (math.sqrt(2 * self.dt) * torch.randn(nbatch, self.hid_dim).to(hidden))
            @ self.sig.T
        )

    """
    calculate the dynamics (score function) 

    Args:
        sample: hidden state
        input: sensory input
        mask: whether to mask the weight (enforce sparsity)
        sym_elec: whether to symmetrize the electric synapse
    """

    def score(self, sample, input, mask=False, sym_elec=False):
        _F = self.cal_F(sample, input, mask, sym_elec)
        # sensory_input = self.Win(input) * torch.sum(input, axis=1).unsqueeze(1)
        # _score[:, self.sensory_mask] += sensory_input
        return torch.linalg.solve(
            self.sig @ self.sig.T + self.get_skew_symmetric(), _F, left=False
        )

    """
    calculate the drift term 

    Args:
        sample: hidden state
        input: sensory input
        mask: whether to mask the weight (enforce sparsity)
        sym_elec: whether to symmetrize the electric synapse
    """

    def cal_F(self, sample, input=None, mask=False, sym_elec=False):
        if mask:
            self.mask_weight()
        if sym_elec:
            W_elec = self.symmetric(self.W_elec)
        else:
            W_elec = self.W_elec
        trans_input = self.W_chem(self.non_lin(sample))
        _F = (
            -sample
            + self.E(trans_input)
            - sample * trans_input  # chemical synapse input
            + sample @ W_elec
            - sample * torch.sum(W_elec, dim=1)  # electric synapse input
            + self.v_rest
        ) * self.gamma
        return _F

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.constant_(m.weight, 0)
                # init.constant_(m.bias, 0)

    # mask the weight matrix according to the connectome
    def mask_weight(self):
        pass
        # W_c = torch.mul(self.sparsity_c, self.magnitudes_c * self.magnitude_scaling_factor_chem)
        # W_e = torch.mul(self.sparsity_e, (self.magnitudes_e + self.magnitudes_e.transpose(0,1)) * self.magnitude_scaling_factor_elec)

    # create a skew-symmetric matrix out of J
    def get_skew_symmetric(self):
        skew_symmetric = self.J - self.J.T
        # skew_symmetric = 0.01 * skew_symmetric / torch.norm(skew_symmetric)
        return skew_symmetric

    @staticmethod
    def symmetric(X):
        return X.triu() + X.triu(1).transpose(-1, -2)


class SparseNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        r_lr: float = 0.1,
        lmda: float = 5e-3,
        maxiter: int = 500,
        device: torch.device = None,
    ):
        super(SparseNet, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.r_lr = r_lr
        self.lmda = lmda
        self.maxiter = maxiter
        # device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        self.U = nn.Linear(hidden_dim, input_dim, bias=False)
        # responses
        self.normalize_weights()

    def inference(self, img_batch):
        r = torch.zeros(
            (img_batch.shape[0], self.hidden_dim),
            requires_grad=True,
            device=self.device,
        )
        converged = False
        # SGD
        optim = torch.optim.SGD([r], lr=self.r_lr)
        # train
        iter = 0
        requires_grad(self.parameters(), False)
        while not converged and iter < self.maxiter:
            old_r = r.clone().detach()
            # pred
            pred = self.U(r)
            # loss
            loss = torch.pow(img_batch - pred, 2).sum(dim=1).mean()
            loss.backward()
            optim.step()
            optim.zero_grad()
            # prox
            r.data = SparseNet.soft_thresholding_(r, self.lmda)
            # convergence
            with torch.no_grad():
                converged = torch.norm(r - old_r) / torch.norm(old_r) < 0.01
            iter += 1
        if iter == self.maxiter:
            print("did not converge")
        requires_grad(self.parameters(), True)
        return r.clone().detach()

    @staticmethod
    def soft_thresholding_(x, alpha):
        with torch.no_grad():
            rtn = F.relu(x - alpha) - F.relu(-x - alpha)
        return rtn.data

    def normalize_weights(self):
        with torch.no_grad():
            self.U.weight.data = F.normalize(self.U.weight.data, dim=0)

    def forward(self, img_batch):
        # inference
        r = self.inference(img_batch)
        # print(np.count_nonzero(r[0].cpu().clone().detach().numpy()) / self.hidden_dim)
        # now predict again
        pred = self.U(r)
        return pred


class Autoencoder(nn.Module):
    def __init__(self, hidden_dim, img_size=28, n_channel=1):
        super(Autoencoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.img_size = img_size
        self.n_channel = n_channel
        # breakpoint()
        self.encoder = nn.Sequential(
            # 1 x 28 x 28/32
            nn.Conv2d(n_channel, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 24 x 24/28
            nn.Conv2d(16, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 20 x 20/24
            nn.Conv2d(16, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 9 x 9/11
            nn.Conv2d(16, 10, kernel_size=4),
            nn.BatchNorm2d(10),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(10 * (img_size//2 -8)**2, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, self.hidden_dim),
        )
        self.decoder = nn.Sequential(
            # 10
            nn.Linear(self.hidden_dim, self.hidden_dim*2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim*2, 10 * 6 * 6),
            nn.ReLU(),
            nn.Unflatten(1, (10, 6, 6)),
            # 10 x 6 x 6
            nn.ConvTranspose2d(10, 16, kernel_size=4),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 9 x 9
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 20 x 20
            nn.ConvTranspose2d(16, 16, kernel_size=5),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            # 16 x 24 x 24
            nn.ConvTranspose2d(16, n_channel, kernel_size=5),
            nn.BatchNorm2d(n_channel),
            nn.ReLU(),
        )
        self.output = nn.ConvTranspose2d(n_channel, n_channel, kernel_size=5)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        if self.n_channel == 3:
            dec = self.output(dec)
        dec = self.sig(dec)
        return dec

class AutoencoderCifar(nn.Module):
    def __init__(self, feature_dim=128):
        super(AutoencoderCifar, self).__init__()
        # Input size: [batch, 3, 32, 32]
        # Output size: [batch, 3, 32, 32]
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),           # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),           # [batch, 48, 4, 4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48*4*4, feature_dim),
# 			nn.Conv2d(48, 96, 4, stride=2, padding=1),           # [batch, 96, 2, 2]
#             nn.ReLU(),
        )
        self.decoder = nn.Sequential(
#             nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),  # [batch, 48, 4, 4]
#             nn.ReLU(),
            nn.Linear(feature_dim, 48*4*4),
            nn.Unflatten(1, (48,4,4)),
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),  # [batch, 24, 8, 8]
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),  # [batch, 12, 16, 16]
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),   # [batch, 3, 32, 32]
            nn.Sigmoid(),
        )

    def forward(self, x, test=True):
        encoded = self.encoder(x)
        if not test:
            x = x + torch.randn_like(x) / 10
        decoded = self.decoder(encoded)
        return decoded