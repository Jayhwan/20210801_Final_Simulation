from parameters import *


def decompose(act_user, time, load_matrix, operator_action, user_action):

    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time))
    elif act_user == 0:
        load_active = np.zeros((1, time))
        load_passive = load_matrix
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    if user_action is not None:
        [x_s, x_b, l] = [user_action[0], user_action[1], user_action[2]]
    else:
        [x_s, x_b, l] = [np.zeros((1, time)), np.zeros((1, time)), np.zeros((1, time))]

    if operator_action is not None:
        [p_s, p_b] = [operator_action[0], operator_action[1]]
    else:
        [p_s, p_b] = [np.zeros((1, time)), np.zeros((1, time))]

    return [x_s, x_b, l, p_s, p_b, load_active, load_passive]


def get_par(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    return np.max(load_total)/np.average(load_total)


def get_ec(act_user, time, load_matrix, operator_action=None, user_action=None):
    if act_user ==0:
        l = np.zeros((1, time))
    else:
        [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    return p_l * np.sum(np.power(load_total, 2))


def operator_objective(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    ec = - get_ec(act_user, time, load_matrix, operator_action, user_action)
    tax = - p_tax * act_user * np.sum(np.power(p_s, 2) + np.power(p_b, 2))

    return ec + tax


def user_objective(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros(1)

    x = np.zeros(act_user)

    for i in range(act_user):
        trans_cost = np.sum(np.multiply(p_s, x_s[i]) - np.multiply(p_b, x_b[i]))
        energy_cost = - p_l * np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        soh_cost = - p_soh * np.sum(np.multiply(x_s[i], np.sum(x_s, axis=0)))
        soh_cost += - p_soh * np.sum(np.multiply(x_b[i], np.sum(x_b, axis=0)))
        x[i] = trans_cost + energy_cost + soh_cost

    return x

def total_user_cost(act_user, time, load_matrix, operator_action=None, user_action=None):
    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)
    if act_user == 0:
        return np.zeros(1)
    x = np.zeros(total_user)
    for i in range(act_user):
        trans_cost = np.sum(np.multiply(p_s, x_s[i]) - np.multiply(p_b, x_b[i]))
        energy_cost = - p_l * np.sum(np.multiply(l[i], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        soh_cost = - p_soh * np.sum(np.multiply(x_s[i], np.sum(x_s, axis=0)))
        soh_cost += - p_soh * np.sum(np.multiply(x_b[i], np.sum(x_b, axis=0)))
        x[i] = trans_cost + energy_cost + soh_cost
    for i in range(act_user, total_user):
        energy_cost = - p_l * np.sum(np.multiply(load_passive[i-act_user], np.sum(l, axis=0) + np.sum(load_passive, axis=0)))
        x[i] = energy_cost
    return -np.sum(x)

def operator_constraint_value_identical(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros((1, time))

    x = np.zeros((2, time))

    x[0] = - p_s
    x[1] = p_s - p_b

    return x


def follower_constraint_value(act_user, time, load_matrix, operator_action=None, user_action=None):

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    if act_user == 0:
        return np.zeros((1, time))

    x = np.zeros((4 + 3 * act_user, time))

    for t in range(time):
        if t == 0:
            x[0, t] = - (q_min + beta_s * np.sum(x_s[:, t]) - beta_b * np.sum(x_b[:, t]))
            x[1, t] = - x[0, t] - q_max
        else:
            x[0, t] = alpha * x[0, t-1] - (beta_s * np.sum(x_s[:, t]) - beta_b * np.sum(x_b[:, t]))
            x[1, t] = - x[0, t] - q_max
        x[2, t] = np.sum(x_s[:, t]) - c_max
        x[3, t] = np.sum(x_b[:, t]) - c_min
        for i in range(act_user):
            x[4+3*i, t] = - x_s[i, t]
            x[4+3*i+1, t] = - x_b[i, t]
            x[4+3*i+2, t] = - l[i, t]
    return x


def follower_constraints_derivative_matrix(act_user, time):

    x = np.zeros((4 + 3 * act_user, time, 2, time))

    for t in range(time):
        if t == 0:
            x[0, t, 0, t] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0, t, 1, t] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1, t] = - x[0, t]
        else:
            x[0, t] = alpha * x[0, t-1]
            x[0, t, 0, t] = act_user * (beta_s * p_l - beta_b * (p_l + p_soh))
            x[0, t, 1, t] = act_user * (beta_s * (p_l + p_soh) - beta_b * p_l)
            x[1, t] = - x[0, t] #########

        x[2, t, 0, t] = - act_user * p_l
        x[2, t, 1, t] = - act_user * (p_l + p_soh)
        x[3, t, 0, t] = - act_user * (p_l + p_soh)
        x[3, t, 1, t] = - act_user * p_l

        for i in range(act_user):
            x[4+3*i, t, 0, t] = p_l
            x[4+3*i, t, 1, t] = p_l + p_soh
            x[4+3*i+1, t, 0, t] = p_l + p_soh
            x[4+3*i+1, t, 1, t] = p_l
            x[4 + 3 * i + 2, t, 0, t] = - p_soh
            x[4 + 3 * i + 2, t, 1, t] = p_soh

    return x/(p_soh*(p_soh+2*p_l))


def almost_same(a, b):
    if np.abs(a-b) <= epsilon:
        return True
    else:
        return False


def direction_finding(act_user, time, load_matrix, operator_action, user_action):

    follower_gradient_matrix = follower_constraints_derivative_matrix(act_user, time)
    follower_constraints_value = follower_constraint_value(act_user, time, load_matrix, operator_action, user_action)

    d = cp.Variable(1)
    r = cp.Variable((2, time))
    v = cp.Variable((2, time))
    g = cp.Variable((4+3*act_user, time))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    objective = d

    constraints = []

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    # First constraint
    constraints += [cp.sum(cp.multiply(2*p_tax*act_user*p_s+2*p_l*act_user*load_total/(p_soh+2*p_l), r[0]))
                    + cp.sum(cp.multiply(2*p_tax*act_user*p_b+2*p_l*act_user*load_total/(p_soh+2*p_l), r[1]))
                    + cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[0]))
                    - cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[1])) <= d]

    # Second constraints
    for t in range(time):
        constraints += [-r[0, t] <= p_s[t] + d]
        constraints += [r[0, t] - r[1, t] <= - p_s[t] + p_b[t] + d]

    # Third constraints
    for t in range(time):
        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*((p_l+p_soh)*v[0, t]+p_l*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*r[0, t]-(p_soh+p_l)*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 0, t], g)) == 0]

        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*(p_l*v[0, t]+(p_l+p_soh)*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*((p_soh+p_l)*r[0, t] - p_l*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 1, t], g)) == 0]

    # Fourth constraints
    for i in range(4+3*act_user):
        for t in range(time):
            if almost_same(follower_constraints_value[i, t], 0):
                constraints += [g[i, t] >= 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t], v)) == 0]
            else:
                constraints += [g[i, t] == 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t] , v)) <= - follower_constraints_value[i, t] + d]

    # Fifth constraint
    constraints += [cp.sum(cp.power(r, 2)) <= 1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS', max_iters=1000)

    return result, d.value, r.value, v.value, g.value


def follower_action_ni(act_user, time, operator_action, load_matrix, init=None):
    load_matrix = load_matrix[:, :time]
    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time))
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]
    p_s = operator_action[0]#.reshape(1, -1)
    p_b = operator_action[1]#.reshape(1, -1)
    gamma = 1000000
    if init is None:
        x_s_tmp = 0 * load_active #np.zeros((active_user, time_step))
        x_b_tmp = 0 * load_active #np.zeros((active_user, time_step))
        l_tmp = x_s_tmp - x_b_tmp + load_active
    else:
        x_s_tmp = init[0]
        x_b_tmp = init[1]
        l_tmp = init[2]
    while True:
        diff=0
        for i in range(act_user):
            t = 1
            #x_s = cp.Variable((act_user, time))
            #x_b = cp.Variable((act_user, time))
            x_s_single = cp.Variable(time)
            x_b_single = cp.Variable(time)
            l_single = x_s_single - x_b_single + load_active[i]
            # new_utility = cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))
            new_utility = 0
            prev_utility = 0
            for k in range(act_user):
                if k == i:
                    new_utility += cp.sum(cp.multiply(p_s, x_s_single) - cp.multiply(p_b, x_b_single))
                    new_utility -= p_l * cp.sum(cp.power(l_single, 2) + cp.multiply(l_single, np.sum(l_tmp, axis=0) - l_tmp[k] + np.sum(load_passive, axis=0)))  ###### 일반적으론 패시브유저도 고려해야함!!
                    new_utility -= p_soh * cp.sum(
                        cp.power(x_s_single, 2) + cp.multiply(x_s_single, np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
                    new_utility -= p_soh * cp.sum(
                        cp.power(x_b_single, 2) + cp.multiply(x_b_single, np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))
                    #prev_utility += np.sum(np.multiply(p_s, x_s_tmp[k]) - np.multiply(p_b, x_b_tmp[k]))
                    #prev_utility -= p_l * np.sum(
                    #    np.power(l_tmp[k], 2) + np.multiply(l_tmp[k], np.sum(l_tmp, axis=0) - l_tmp[k]))
                    #prev_utility -= p_soh * np.sum(
                    #    np.power(x_s_tmp[k], 2) + np.multiply(x_s_tmp[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
                    #prev_utility -= p_soh * np.sum(
                    #    np.power(x_b_tmp[k], 2) + np.multiply(x_b_tmp[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))
            # prev_utility = np.sum(np.multiply(a_o[0], x_s_tmp) - np.multiply(a_o[1], x_b_tmp))\
            #          - p_l * np.sum(np.power(np.sum(l_tmp, axis=0), 2) + np.multiply(np.sum(l_tmp, axis=0), np.sum(load_passive, axis=0))) \
            #          - p_soh * np.sum(np.power(np.sum(x_s_tmp, axis=0), 2) + np.power(np.sum(x_b_tmp, axis=0), 2))
            difference = gamma * cp.sum(cp.power(cp.vstack([x_s_single, x_b_single]) - np.vstack([x_s_tmp[i], x_b_tmp[i]]), 2)) / 2
            utility = new_utility - prev_utility - difference
            constraints = []
            constraints += [l_single >= 0]
            constraints += [x_s_single >= 0]
            constraints += [x_b_single >= 0]
            ess_matrix = np.fromfunction(np.vectorize(lambda i, j: 0 if i < j else np.power(alpha, i - j)),
                                         (time, time), dtype=float)

            q_ess = q_min * np.fromfunction(np.vectorize(lambda i, j: np.power(alpha, i - j + 1)), (time, act_user),
                                                  dtype=float) \
                          + beta_s * ess_matrix @ (np.sum(x_s_tmp, axis=0)-x_s_tmp[i]+x_s_single).T - beta_b * ess_matrix @ (np.sum(x_b_tmp, axis=0)-x_b_tmp[i]+x_b_single).T
            constraints += [q_min <= q_ess, q_ess <= q_max]
                # if time_step == 1:
                #    for j in range(active_user):
                #        constraints += [- x_s[j][0] + x_b[j][0] - np.sum(x_s_tmp) + x_s_tmp[j][0] + np.sum(x_b_tmp) - x_b_tmp[j][0] <= 0]
                # else:
                #    constraints += [q_min <= q_ess, q_ess <= q_max]
            constraints += [np.sum(x_s_tmp, axis=0)-x_s_tmp[i]+x_s_single <= c_max]
            constraints += [np.sum(x_b_tmp, axis=0)-x_b_tmp[i]+x_b_single <= c_min]
            prob = cp.Problem(cp.Maximize(utility), constraints)
            start = timer.time()
            result = prob.solve(solver='ECOS')
            end = timer.time()
            if i%50 == 0:
                print(i)
                print("### ni ###")
                #print(prev_utility)
                print("EC  :", get_ec(act_user, time, load_matrix, operator_action, np.array([x_s.value, x_b.value, l.value])))
                print(difference.value)
                #print("x_s :", x_s.value.reshape(act_user, -1))
                #print("x_b :", x_b.value.reshape(act_user, -1))
                #print(i, "l   :", l.value[0])#.reshape(act_user, -1))
            x_s_tmp[i] = (1-t)*x_s_tmp[i] + t*x_s_single.value
            x_b_tmp[i] = (1-t)*x_b_tmp[i] + t*x_b_single.value
            l_tmp[i] = (1-t)*l_tmp[i] + t*l_single.value
            diff += difference.value
        if diff < 2*1e-7:
            break

    return result, x_s_tmp, x_b_tmp, l_tmp, end - start

def direction_finding_ni(act_user, time, load_matrix, operator_action, user_action):

    follower_gradient_matrix = follower_constraints_derivative_matrix(act_user, time)
    follower_constraints_value = follower_constraint_value(act_user, time, load_matrix, operator_action, user_action)

    d = cp.Variable(1)
    r = cp.Variable((2, time))
    v = cp.Variable((2, time))
    g = cp.Variable((4+3*act_user, time))

    [x_s, x_b, l, p_s, p_b, load_active, load_passive] = decompose(act_user, time, load_matrix, operator_action, user_action)

    objective = d

    constraints = []

    load_total = np.sum(l, axis=0) + np.sum(load_passive, axis=0)

    # First constraint
    constraints += [cp.sum(cp.multiply(2*p_tax*act_user*p_s+2*p_l*act_user*load_total/(p_soh+2*p_l), r[0]))
                    + cp.sum(cp.multiply(2*p_tax*act_user*p_b+2*p_l*act_user*load_total/(p_soh+2*p_l), r[1]))
                    + cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[0]))
                    - cp.sum(cp.multiply(2*act_user*p_l/(p_soh+2*p_l)*load_total, v[1])) <= d]

    # Second constraints
    for t in range(time):
        constraints += [-r[0, t] <= p_s[t] + d]
        constraints += [r[0, t] - r[1, t] <= - p_s[t] + p_b[t] + d]

    # Third constraints
    for t in range(time):
        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*((p_l+p_soh)*v[0, t]+p_l*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*(p_l*r[0, t]-(p_soh+p_l)*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 0, t], g)) == 0]

        constraints += [2*act_user*act_user/(p_soh*(p_soh+2*p_l))*(p_l*v[0, t]+(p_l+p_soh)*v[1, t])
                        - act_user*(2*act_user-1)/(p_soh*(p_soh+2*p_l))*((p_soh+p_l)*r[0, t] - p_l*r[1, t])
                        + cp.sum(cp.multiply(follower_gradient_matrix[:, :, 1, t], g)) == 0]

    # Fourth constraints
    for i in range(4+3*act_user):
        for t in range(time):
            if almost_same(follower_constraints_value[i, t], 0):
                constraints += [g[i, t] >= 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t], v)) == 0]
            else:
                constraints += [g[i, t] == 0]
                constraints += [cp.sum(cp.multiply(follower_gradient_matrix[i, t] , v)) <= - follower_constraints_value[i, t] + d]

    # Fifth constraint
    constraints += [cp.sum(cp.power(r, 2)) <= 1]

    prob = cp.Problem(cp.Minimize(objective), constraints)
    result = prob.solve(solver='ECOS', max_iters=1000)

    return result, d.value, r.value, v.value, g.value


def step_size_ni(act_user, time, load_matrix, operator_action, user_action, d, r):

    update_coef = 0.5
    s = 0.95
    for i in range(2000):
        next_operator_action = operator_action + s * r
        result, x_s, x_b, l, taken_time = follower_action_ni(act_user, time, next_operator_action, load_matrix)
        next_follower_action = np.array([x_s, x_b, l])
        update = True
        if operator_objective(act_user, time, load_matrix, next_operator_action, next_follower_action)\
            >= operator_objective(act_user, time, load_matrix, operator_action, user_action) - update_coef * s * d:
            if np.any(operator_constraint_value_identical(act_user, time, load_matrix, next_operator_action, next_follower_action) > 0):
                update = False
        else:
            update = False

        if update:
            return s, next_operator_action, next_follower_action
        else:
            s *= update_coef
            if s <= 1e-7:
                break
    return 0, operator_action, user_action


def iterations_ni(act_user, time, load_matrix, operator_action, init=None):

    a_o = operator_action
    result, x_s, x_b, l, taken_time = follower_action_ni(act_user, time, a_o, load_matrix, init)

    a_f = np.array([x_s, x_b, l])

    min_step_size = 1e-6

    print(get_ec(act_user, time, load_matrix, a_o, a_f))
    print(get_par(act_user, time, load_matrix, a_o, a_f))

    for i in range(max_iter):
        print("ITER :", i)
        print("DIRECTION")
        result, d, r, v, g = direction_finding_ni(act_user, time, load_matrix, a_o, a_f)
        print("d :", result)
        #print("r :", r)
        print("STEP")
        b, next_a_o, next_a_f = step_size_ni(act_user, time, load_matrix, a_o, a_f, d, r)
        if b != 0:
            print("s :", b)
            a_o = next_a_o
            a_f = next_a_f
        else:
            print("NO UPDATE")
            return a_o, a_f

        print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
        print("PAR :", get_par(act_user, time, load_matrix, a_o, a_f))
        if b <= min_step_size:
            break

    return a_o, a_f

