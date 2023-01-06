from models import LIF
from utility import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = use_gpu()
    hid_dim= 16
    model = LIF(hid_dim, .1).to(device)
    load("./model/best_model_var0.5", model)
    traj = gen_traj(model,torch.zeros([1,model.hid_dim]).to(device), 6000).detach().numpy()
    # print(traj)
    # plt.plot(traj[100:, 0], traj[100:, 1],'.')
    plt.hist(traj[1000:, 0], bins=100)
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    plt.show()