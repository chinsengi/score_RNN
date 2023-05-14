
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

'''
vanilla synaptic current model of RNN, without output layer
'''
class SynCurrentDyn(torch.nn.Module):
    def __init__(self, hid_dim, dt=0.001):
        super().__init__()
        self.hid_dim = hid_dim
        self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Identity()
        self.non_lin = nn.ReLU()
        self.dt = dt
        self.is_set_weight=True

    def forward(self, input):
        v = self.score(input)
        # mean = torch.zeros(self.hid_dim-1).to(input)-2
        # var  = torch.ones(self.hid_dim-1).to(input)/2
        # GMM_mean = torch.tensor([-2, 2 ]).to(input).unsqueeze(0)
        # GMM_var = torch.tensor([0.5, .5]).to(input).unsqueeze(0)
        # v = torch.cat((score_GMM(input[:,[0]], GMM_mean, GMM_var), score_normal(input[:,1:], mean, var)), 1)
        # v = score_normal(input,mean, var)
        return input + self.dt*v + math.sqrt(2*self.dt)*torch.randn_like(input)

    def score(self, input): 
        input_trans = self.non_lin(input)
        v = self.W(input_trans)
        # v = v - torch.diag(self.W.weight)*input_trans
        # v = self.W(input)
        # v = self.W(torch.tanh(v))
        # v = v - torch.diag(self.W2.weight)*torch.tanh(v)
        v = v - self.gamma.T*input
        return v
    
    def set_weight(self):
        pass

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)

'''
vanilla synaptic current & firing rate model of RNN, without output layer
'''
class NeuralDyn(torch.nn.Module):
    def __init__(self, hid_dim, synap=True, dt=0.001, non_lin=nn.ReLU()):
        super().__init__()
        self.hid_dim = hid_dim
        self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.sig = Parameter(torch.zeros(hid_dim, hid_dim, requires_grad=True))
        self.W = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Identity()
        self.non_lin = non_lin
        self.dt = dt
        self.synap = synap # if the dynamics is synaptic current or not (firing rate dynamics)
        self.is_set_weight=True

    def forward(self, input):
        v = self.score(input)
        return input + self.dt*v + math.sqrt(2*self.dt)*torch.randn_like(input)

    def score(self, input): 
        if self.synap:
            input_trans = self.non_lin(input)
            v = self.W(input_trans)
        else:
            input_trans = self.W(input)
            v = self.non_lin(input_trans)
        v = v - self.gamma.T*input
        if not self.synap:
            v = v@self.sig@self.sig.T
        return v
    
    def set_weight(self):
        pass

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)

class rand_RNN(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, in_dim=3, dt=0.001, non_lin=nn.ReLU()):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        # self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Linear(hid_dim, out_dim, bias = False)
        self.W1 = nn.Linear(hid_dim, out_dim, bias=False)
        self.W2 = nn.Linear(out_dim, hid_dim, bias = True)
        self.is_set_weight = False
        self.non_lin = non_lin
        # self.non_lin = nn.LeakyReLU(0.1)
        # self.non_lin = torch.nn.Tanh()
        self.dt = dt

        self.Win = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, out_dim),
        )
        self.Win_rec = nn.Sequential(
            nn.Linear(in_dim, hid_dim*2),
            nn.ReLU(),
            nn.Linear(hid_dim*2, hid_dim),
        )


    def forward(self, hidden, input=None):
        v = self.cal_v(hidden, input)
        nbatch = hidden.shape[0]
        return hidden + self.dt*v + (math.sqrt(2*self.dt)*torch.randn(nbatch, self.out_dim).to(hidden))@self.sig.T
    
    def set_weight(self):
        W_rec_tilde = self.W2.weight
        self.W_out.weight = self.W1.weight
        self.sig = torch.linalg.solve(self.W1.weight@\
            self.W1.weight.T, self.W1.weight.T, left=False)
        self.W_rec.weight = Parameter(W_rec_tilde@self.W1.weight)
        self.W_rec.bias = Parameter(self.W2.bias)
        self.is_set_weight = True

    def cal_v(self, hidden, input=None):
        if input is not None:
            input_rec = self.Win_rec(input)
        else:
            input_rec = 0  
        v = -hidden + self.non_lin(self.W_rec(hidden)+input_rec)
        return v      

    def score(self, sample, input=None):
        if input is not None:
            input_out = self.Win(input)
            input_rec = self.Win_rec(input)
        else:
            input_out = 0
            input_rec = 0
        internal_score = -sample + self.W1(self.non_lin(self.W2(sample) + input_rec))
        return internal_score + input_out
    
    def true_input(self, x):
        x = self.Win(x)
        wout = self.W_out.weight
        return (0.5*wout@wout.T@x.T).T
    
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight)
            # init.constant_(m.bias, 0)



class RNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.w_in = nn.Linear(in_dim, hid_dim, bias=True)
        self.w_out = nn.Linear(hid_dim, out_dim, bias=True)
        self.w_hh = nn.Linear(hid_dim, hid_dim, bias=True)
    
    def forward(self,initial_state):
        #TODO add input here
        next = self.w_hh(initial_state)
        next = torch.tanh(next)
        return next

    # calculate the log determinant of the function at hidden_state
    def cal_logdet(self, hidden_state):
        batch_size = hidden_state.shape[0]
        hid_dim = hidden_state.shape[1]
        ans = torch.zeros(batch_size, 1).to(hidden_state)
        for i in range(batch_size):
            jac = jacobian(self.forward, hidden_state[i], create_graph=True) 
            ans[i] = torch.linalg.slogdet(jac)[1]
        return ans

class SparseNet(nn.Module):

    def __init__(self, input_dim:int, hidden_dim:int, r_lr:float=0.1, lmda:float=5e-3, maxiter:int=500, device:torch.device=None):
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
        r = torch.zeros((img_batch.shape[0], self.hidden_dim), requires_grad=True, device=self.device)
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
