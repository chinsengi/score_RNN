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

if __name__ == '__main__':
    device = use_gpu()
    hid_dim, out_dim= 128, 1
    samples = (torch.rand([1000, hid_dim])*10-5).to(device)
    n_level = 10
    noise_levels = [1/math.exp(math.log(10)*n/n_level) for n in range(n_level)]
    for i in range(n_level):
        model = rand_RNN(hid_dim, out_dim).to(device)
        model.dt = 1e-5*(noise_levels[i]/noise_levels[-1])**2
        load(f"./model/rand_RNN_ep{i*20}", model)
        samples = gen_sample(model,samples, 100)
        plt.figure()
        plt.hist(model.W_out(samples).detach().cpu().numpy())
    samples = model.W_out(samples).detach().cpu().numpy()
    plt.figure()
    plt.hist(samples)
    # plt.hist(torch.randn(1000)*math.sqrt(0.5))
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    plt.show()

# %%
