import torch
import os
from math import inf
import math
from tqdm import tqdm

# load a model
def load(path, model):
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))

# create directory
def create_dir(path='./model'):
    isExist = os.path.exists(path)
    if not isExist:
        os.mkdir(path)

# score for normal distribution
def score_normal(h, mean, variance):
    return (mean - h)/variance

def div_score_normal(variance):
    return -torch.sum(1/variance)

def normal_pdf(x, mean, variance):
    return torch.exp(-(x-mean)**2/(2*variance))\
                /(torch.sqrt(2*torch.pi*variance))

# score for 1-D Gaussian mixture distribution
def score_GMM(x, mean, variance):
    n_comp = mean.shape[1]
    pdf_vec = normal_pdf(x, mean, variance)/n_comp
    pdf = torch.sum(pdf_vec, 1, keepdim=True)
    return torch.sum(-pdf_vec*(x-mean)/variance,1, keepdim=True)/pdf

def div_score_GMM(x, mean, variance):
    n_comp = mean.shape[1]
    pdf_vec = normal_pdf(x, mean, variance)/n_comp
    pdf = torch.sum(pdf_vec, 1)
    dpdf_vec = -pdf_vec*(x-mean)/variance
    ddpdf_vec = -pdf_vec/variance - dpdf_vec*(x-mean)/variance
    return (torch.sum(ddpdf_vec, 1)*pdf - torch.sum(dpdf_vec, 1)**2)/pdf**2

def train_normal(model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    # m = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    hid_dim = model.hid_dim
    mean = torch.zeros(hid_dim, device=device)+1
    var  = torch.ones(hid_dim, device=device)/2
    min_loss = inf
    nepoch = 60
    for epoch in range(nepoch):
        for batchId, h in enumerate(loader):
            # print(batchId)
            h = h.to(device)
            div_f = div_score_normal(var)
            f = score_normal(h, mean, var)
            loss = 0.5*torch.norm(div_f + torch.inner(f, f) + torch.sum(model.gamma) - torch.inner(f, model.cal_v(h)))

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batchId % 100 == 0:
            #     print(f"loss: {loss.item():>7f}, batchId: {batchId}")
            if batchId % 10 == 0:
                if(loss.item()<min_loss):
                    torch.save(model.state_dict(), f"./model/best_model_var{var[0].item():.1f}")
        print(f"loss: {loss.item():>7f}, Epoch: {epoch}")

'''
train RNN to produce Gaussian mixture distribution with analytic score matching
'''
def train_GMM(model, loader, GMM_mean, GMM_var, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    # m = MultivariateNormal(torch.zeros(2).to(device), torch.eye(2).to(device))
    hid_dim = model.hid_dim
    mean = torch.zeros(hid_dim-1, device=device)
    var  = torch.ones(hid_dim-1, device=device)/2
    # mean_test = torch.zeros(hid_dim, device=device)
    # var_test  = torch.ones(hid_dim, device=device)/2
    min_loss = inf
    nepoch = 50
    for epoch in range(nepoch):
        for batchId, h in enumerate(loader):
            # print(batchId)
            h = h.to(device)
            div_f = div_score_GMM(h[:, [0]], GMM_mean, GMM_var)+div_score_normal(var)
            # assert(torch.norm(div_f-div_score_normal(var_test))<0.001)
            f = torch.cat((score_GMM(h[:, [0]], GMM_mean, GMM_var), score_normal(h[:,1:], mean, var)), 1)
            # assert(torch.norm(f-score_normal(h, mean_test, var_test))<0.001)
            loss = 0.5*torch.norm(div_f + torch.inner(f, f) + torch.sum(model.gamma) - torch.inner(f, model.cal_v(h)))**2

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batchId % 100 == 0:
            #     print(f"loss: {loss.item():>7f}, batchId: {batchId}")
        if(loss.item()<min_loss):
            torch.save(model.state_dict(), f"./model/best_model_var{var[0].item():.1f}")
            min_loss = loss.item()
            print(f"min_loss = {loss.item()}")
        print(f"loss: {loss.item():>7f}, Epoch: {epoch}")

'''
train Gaussian mixture model with conditional noise
'''
def train_GMM_cn(model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
    min_loss = inf

    # annealing noise
    n_level = 20
    noise_levels = [10/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

    nepoch = 200
    model.train()
    for epoch in tqdm(range(nepoch)):
        if epoch % (nepoch//n_level) ==0:
            noise_level = noise_levels[epoch//(nepoch//n_level)]
            print(f"noise level: {noise_level}")
            torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_ep{epoch}")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        for batchId, h in enumerate(loader):
            # print(batchId)
            h = h.to(device)
            h_noisy = h + torch.randn_like(h)*noise_level
            loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # if batchId % 100 == 0:
            #     print(f"loss: {loss.item():>7f}, batchId: {batchId}")
        print(f"loss: {loss.item():>7f}, Epoch: {epoch}")
    torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_ep{nepoch}")    

def train_MNIST(model, loader, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=0.001)
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=.9)

    # annealing noise
    n_level = 10
    noise_levels = [1/math.exp(math.log(100)*n/n_level) for n in range(n_level)]

    nepoch = 400
    model.train()
    for epoch in tqdm(range(nepoch)):
        if epoch % (nepoch//n_level) ==0:
            noise_level = noise_levels[epoch//(nepoch//n_level)]
            print(f"noise level: {noise_level}")
            torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_MNIST_ep{epoch}")
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.001)
        for h in loader:
            # print(batchId)
            h = h[0].to(device, torch.float32)
            h_noisy = h + torch.randn_like(h)*noise_level
            loss = 0.5*((model.score(h_noisy) - (h - h_noisy)/noise_level**2)**2).sum(dim=-1).mean(dim=0)

            #backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"loss: {loss.item():>7f}, Epoch: {epoch}")
    torch.save(model.state_dict(), f"./model/{model.__class__.__name__}_MNIST_ep{nepoch}")   


def gen_traj(model, initial_state, length):
    model.eval()
    nbatch = initial_state.shape[0]
    hidden_list = torch.zeros(length, nbatch, model.hid_dim)
    hidden_list[0] = initial_state
    next = initial_state
    for i in range(1, length):
        next = model(next)
        hidden_list[i] = next
    return hidden_list

'''
generate samples from a langevin system. 

param:
    length: number of steps to generate the sample
'''
def gen_sample(model, initial_state, length):
    next = initial_state
    for i in range(length):
        next = model(next)
    return next

def use_gpu():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")
    return device