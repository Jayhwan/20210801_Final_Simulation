import matplotlib.pyplot as plt
import numpy as np
from common_function import *
from indiv_function import *
from scipy.stats import truncnorm

loads = np.load("load_single_low_var.npy", allow_pickle=True)
ec_indiv_ni = []
ec_indiv = []
ec_share = []
ec_share_ni = []
par_indiv_ni = []
par_indiv = []
par_share = []
par_share_ni = []
util_indiv_ni = []
util_indiv = []
util_share = []
util_share_ni = []

a_o = 1.5*np.random.random((2, time_step)) + 0.5 * np.ones((2, time_step))
tmp = a_o[0]
a_o[0] = np.minimum(a_o[0], a_o[1])
a_o[1] = np.maximum(tmp, a_o[1])
np.save("2_a_o_tmp.npy", a_o)
for i in range(1, 11):
    print(i)
    load = loads[:total_user, :time_step]
    #result, x_s, x_b, l, taken_time = iterations(i, time_step, load, a_o)
    a_o, a_f = iterations(i, time_step, load, a_o)
    ec_share += [get_ec(i, 12, load, a_o, a_f)]
    par_share += [get_par(i, 12, load, a_o, a_f)]
    #util_share += [total_user_cost(i, 12, load, np.ones((2, 12))), np.array([x_s, x_b, l])]

    #result, x_s, x_b, l, taken_time = iterations_ni(i, time_step, load, a_o)
    a_o, a_f = iterations_ni(i, time_step, load, a_o)
    ec_share_ni += [get_ec(i, 12, load, a_o, a_f)]
    par_share_ni += [get_par(i, 12, load, a_o, a_f)]
    #util_share_ni += [total_user_cost(i, 12, load, np.ones((2, 12))), np.array([x_s, x_b, l])]

    #result, x_s, x_b, l, taken_time = iterations_indiv_ni(i, time_step, load, a_o)
    a_o, a_f = iterations_indiv_ni(i, time_step, load, a_o)
    ec_indiv_ni += [get_ec(i, 12, load, a_o, a_f)]
    par_indiv_ni += [get_par(i, 12, load, a_o, a_f)]
    #util_indiv_ni += [total_user_cost(i, 12, load, np.ones((2, 12))), np.array([x_s, x_b, l])]

    #result, x_s, x_b, l, taken_time = iterations_indiv(i, time_step, load, a_o)
    a_o, a_f = iterations_indiv(i, time_step, load, a_o)
    ec_indiv += [get_ec(i, 12, load, a_o, a_f)]
    par_indiv += [get_par(i, 12, load, a_o, a_f)]
    #util_indiv += [total_user_cost(i, 12, load, np.ones((2, 12))), np.array([x_s, x_b, l])]

    np.save("2_ec_data_3.npy", [ec_share, ec_share_ni, ec_indiv, ec_indiv_ni])
    np.save("2_par_data_3.npy", [par_share, par_share_ni, par_indiv, par_indiv_ni])
    #np.save("2_util_data_1.npy", [util_share, util_share_ni, util_indiv, util_indiv_ni])

[ec_share, ec_share_ni, ec_indiv, ec_indiv_ni] = np.load("2_ec_data_3.npy", allow_pickle=True)
#np.save("data3.npy", [ec_share, ec_share_ni, ec_indiv, ec_indiv_ni])
plt.plot(ec_indiv_ni, label='indiv ni')
plt.plot(ec_indiv, label='indiv')
plt.plot(ec_share, label='share')
plt.plot(ec_share_ni, label='share ni')

plt.legend()
plt.show()