import matplotlib.pyplot as plt
import numpy as np
from time_function import *
from indiv_function import *
from scipy.stats import truncnorm

a_o = np.ones((2, time_step))
load = np.array([[1,2,3],[1,3,2],[2,1,1]])
#load = np.random.random((total_user, time_step))
result, x_s, x_b, l, data = follower_action_ni_time(3, 3, a_o, load)

plt.plot(np.sum(load, axis=0), color='k', label='req')
plt.plot(np.sum(l, axis=0), color='r', label='ni')
plt.legend()
plt.show()