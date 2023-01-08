# %%
from models import LIF
from utility import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    device = use_gpu()
    hid_dim= 16
    model = LIF(hid_dim, .1).to(device)
    load("./model/best_model_ep200", model)
    traj = gen_sample(model,(torch.rand([1000,hid_dim])*10-5).to(device), 1000)\
        .detach().cpu().numpy()
    # tmp = torch.zeros([1,hid_dim]).to(device)
    # tmp[0] = 2
    # print(model.cal_v(tmp)[:,0])
    # plt.plot(traj[100:, 0], traj[100:, 1],'.')
    print(traj)
    plt.hist(traj[:, 0])
    plt.hist(torch.randn(1000)*math.sqrt(0.5))
    # plt.xlim([-3, 3])
    # plt.ylim([-3, 3])
    plt.show()
# %%
