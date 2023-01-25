
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
import math
from torch.autograd.functional import jacobian
from utility import *


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

class FR(torch.nn.Module):
    def __init__(self, hid_dim, dt=0.001):
        super().__init__()
        self.hid_dim = hid_dim
        self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W2 = nn.Linear(hid_dim, hid_dim, bias=True)
        # self.W_out = nn.Linear(hid_dim, out_dim, bias = True)
        self.non_lin = torch.tanh
        self.dt = dt

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
        v = v - torch.diag(self.W.weight)*input_trans
        # v = self.W(input)
        # v = self.W(torch.tanh(v))
        # v = v - torch.diag(self.W2.weight)*torch.tanh(v)
        v = v - self.gamma.T*input
        return v

class rand_RNN(torch.nn.Module):
    def __init__(self, hid_dim, out_dim, dt=0.001):
        super().__init__()
        self.hid_dim = hid_dim
        self.out_dim = out_dim
        self.gamma = Parameter(torch.ones(hid_dim, 1, requires_grad=True))
        self.W_rec = nn.Linear(hid_dim, hid_dim, bias=True)
        self.W_out = nn.Linear(hid_dim, out_dim, bias = False)
        self.W1 = nn.Linear(hid_dim, out_dim, bias=False)
        self.W2 = nn.Linear(out_dim, hid_dim, bias = True)
        self.is_set_weight = False
        self.non_lin = torch.relu
        self.dt = dt

    def forward(self, input):
        v = self.cal_v(input)
        return input + self.dt*v/2 + math.sqrt(self.dt)*torch.randn_like(input)

    def set_weight(self):
        W_rec_tilde = self.W2.weight
        self.W_out.weight = Parameter(torch.linalg.solve(self.W1.weight@\
            self.W1.weight.T, self.W1.weight))
        self.W_rec.weight = Parameter(W_rec_tilde@self.W_out.weight)
        self.W_rec.bias = Parameter(self.W2.bias)
        self.is_set_weight = True

    def cal_v(self, input):  
        v = self.non_lin(self.W_rec(input))
        return v      

    def score(self, input):
        return self.W1(self.non_lin(self.W2(input)))

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