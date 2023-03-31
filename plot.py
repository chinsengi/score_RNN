############################ Pending Deprecation ############################
# %%
from models import FR
from utility import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = use_gpu()
    hid_dim= 16
    model = FR(hid_dim, .01).to(device)
    load("./model/best_model_ep190", model)
    traj = gen_sample(model,(torch.rand([1000,hid_dim])*10).to(device), 1000)\
        .detach().cpu().numpy()
    tmp = torch.zeros([1,hid_dim]).to(device)
    tmp[0] = 2
    print(model.cal_v(tmp)[:,0])
    # plt.plot(traj[100:, 0], traj[100:, 1],'.')
    # print(traj)
    plt.hist(traj[:, 0])
    # plt.hist(torch.randn(1000)*math.sqrt(0.5))
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    plt.show()

# %%
from models import rand_RNN
from utility import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    device = use_gpu()
    hid_dim, out_dim= 512, 1
    samples = (torch.rand([1000, hid_dim])*4-2).to(device)
    n_level = 20
    noise_levels = [10/math.exp(math.log(100)*n/n_level) for n in range(n_level)]
    with torch.no_grad():
        for i in range(1,n_level+1):
            model = rand_RNN(hid_dim, out_dim).to(device)
            load(f"./model/rand_RNN_ep{i*10}", model)
            model.set_weight()
            tmp = model.score(torch.arange(-5, 5, .1).to(device).reshape(1, -1,1 )).squeeze().detach().cpu().numpy()
            plt.plot(np.arange(-5,5,.1), tmp)
            model.dt = 1e-2
            samples = gen_sample(model, samples, 1000)
            # plt.figure()
            # plt.hist(model.W_out(samples).detach().cpu().numpy(), bins=50)
    plt.figure()
    samples = model.W_out(samples).detach().cpu().numpy()
    plt.hist(samples, bins=50)
    plt.show()

# %%
