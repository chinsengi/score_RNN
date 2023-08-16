import numpy as np
import matplotlib.pyplot as plt
from utility import *
import scienceplots
plt.style.use("science")
if __name__ == "__main__":
    first_normal = np.load("./image/LAP/8_SR_wass_dist_first.npy")
    first_fast = np.load("./image/LAP/1_SR_wass_dist_first.npy")
    xt = np.arange(0, len(first_fast), 1)
    plt.plot(xt, first_fast, label='Accelerated')
    plt.plot(xt, first_normal, label='Langevin')
    plt.legend()
    plt.xlabel('time (ms)')
    plt.ylabel('Wasserstein distance')
    savefig('./image/LAP', 'SR_wass_dist_first', 'png')

    plt.figure()
    sec_normal = np.load("./image/LAP/8_SR_wass_dist_sec.npy")
    sec_fast = np.load("./image/LAP/1_SR_wass_dist_sec.npy")
    plt.plot(xt, sec_fast, label='Accelerated')
    plt.plot(xt, sec_normal, label='Langevin')
    plt.xlabel('time (ms)')
    plt.ylabel('Wasserstein distance')
    plt.legend()
    savefig('./image/LAP', 'SR_wass_dist_sec', 'png')