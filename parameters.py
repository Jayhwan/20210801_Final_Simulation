import numpy as np
import cvxpy as cp
import time as timer
import matplotlib.pyplot as plt

total_user = 100
time_step = 12
max_iter = 300
alpha = 0.9956
beta_s = 0.99
beta_b = 1.01

p_soh = 0.010
p_l = 1

p_tax = 0.00001

q_max = 100.
q_min = 0.

c_max = 10.
c_min = 10.

epsilon = 5e-5
