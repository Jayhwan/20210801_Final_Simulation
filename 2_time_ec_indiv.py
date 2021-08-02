import matplotlib.pyplot as plt
import numpy as np
from time_function import *
from indiv_function import *
from scipy.stats import truncnorm

#load = np.load("one_type_load.npy", allow_pickle=True)[:total_user, :time_step]
load = np.load("three_type_load.npy", allow_pickle=True)[:total_user, :time_step]
a_o =5 * np.random.random((2, time_step))# + 0.5 * np.ones((2, time_step))
tmp = a_o[0]
a_o[0] = np.minimum(a_o[0], a_o[1])
a_o[1] = np.maximum(tmp, a_o[1])
a_o[0] /= 1
#a_o = np.ones((2, time_step))
np.save("1_a_o_tmp.npy", a_o)
a_o = np.ones((2, time_step))
print("##################")
#a_o = np.load("1_a_o_tmp.npy", allow_pickle=True)[:,:time_step]
for i in range(51):
    print("act user :", i)
    o, f, data2 = iterations_ni_indiv_time(i, time_step, load, a_o)
    #if i!= 0:
        #print("check")
        #load = np.load("three_type_load.npy", allow_pickle=True)[:total_user, :time_step]
        #result, d, r, v, g = direction_finding(i, time_step, load, o, f)
        #b, o, f = step_size_ni_time(i, time_step, load, o, f, d, r)
        #print("last step size :", b)
    np.save("1_time_ec_indiv_"+str(i)+".npy", data2)

