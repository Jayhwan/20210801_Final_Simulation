import matplotlib.pyplot as plt
import numpy as np
from time_function import *
from indiv_function import *
from scipy.stats import truncnorm

#data1 = np.load("share_time_tmp.npy", allow_pickle=True)
data2 = np.load("share_ni_time_tmp.npy", allow_pickle=True)
#data1 = np.load("1_time_ec_kkt_30_1.npy", allow_pickle=True)
#data2 = np.load("1_time_ec_ni_30_1.npy", allow_pickle=True)
load = np.load("two_type_load.npy", allow_pickle=True)[:total_user, :time_step]

x2 = np.array(data2)
t2 = data2[:, 0, 0]
ec2 = []
#plt.plot(np.sum(load, axis=0), color='k', linewidth=3)
#print(data2[0, 1])
#print(data2[-1, 1])
for i in range(len(x2)):
    ec2 += [get_ec(2, time_step, load, data2[i, 1], data2[i, 2])]
    #if i%5==0:
    #    plt.plot(np.sum(data2[i, 2][2], axis=0)+np.sum(load[2:], axis=0), label=str(i))
#plt.legend()
#plt.show()
#print(t2)
#t2 = t2 - np.ones(len(t2)) * t2[0]

x = 0
print(t2, ec2)
plt.plot(t2[x:], ec2[x:], label='ni')
plt.legend()
plt.show()