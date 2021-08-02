from common_function import *


def follower_action_ni_time(act_user, time, operator_action, load_matrix, start=None, data=None, init=None, gamma=None):
    if total_user == act_user:
        load_active = load_matrix
        load_passive = np.zeros((1, time))
    else:
        load_active = load_matrix[:act_user, :]
        load_passive = load_matrix[act_user:, :]

    p_s = operator_action[0]#.reshape(1, -1)
    p_b = operator_action[1]#.reshape(1, -1)
    if gamma is None:
        gamma = 1
    else:
        gamma = 1
    if init is None:
        x_s_tmp = np.zeros((act_user, time))
        x_b_tmp = np.zeros((act_user, time))
        l_tmp = x_s_tmp - x_b_tmp + load_active
    else:
        x_s_tmp = init[0]
        x_b_tmp = init[1]
        l_tmp = init[2]
    if start is not None:
        a_o = operator_action
        a_f = np.array([x_s_tmp, x_b_tmp, l_tmp])
        cur_time = timer.time() - start
        data += [[cur_time, a_o, a_f]]
        np.save("share_ni_time_tmp.npy", data)
    #print("EC   :", get_ec(act_user, time, load_matrix, operator_action, np.array([x_s_tmp, x_b_tmp, l_tmp])))
    iter = 0
    while True:
        diff=0
        t= 1#/(1+iter)
        for i in range(act_user):
            #x_s = cp.Variable((act_user, time))
            #x_b = cp.Variable((act_user, time))
            x_s_single = cp.Variable(time)
            x_b_single = cp.Variable(time)
            l_single = x_s_single - x_b_single + load_active[i]
            #l = x_s - x_b + load_active
            # new_utility = cp.sum(cp.multiply(a_o[0], x_s) - cp.multiply(a_o[1], x_b))
            new_utility = 0
            prev_utility = 0
            for k in range(act_user):
                if k == i:
                    new_utility += cp.sum(cp.multiply(p_s, x_s_single) - cp.multiply(p_b, x_b_single))
                    new_utility -= p_l * cp.sum(cp.power(l_single, 2) + cp.multiply(l_single, np.sum(l_tmp, axis=0) - l_tmp[
                        k] + np.sum(load_passive, axis=0)))  ###### 일반적으론 패시브유저도 고려해야함!!
                    new_utility -= p_soh * cp.sum(
                        cp.power(x_s_single, 2) + cp.multiply(x_s_single, np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
                    new_utility -= p_soh * cp.sum(
                        cp.power(x_b_single, 2) + cp.multiply(x_b_single, np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))
                    prev_utility += np.sum(np.multiply(p_s, x_s_tmp[k]) - np.multiply(p_b, x_b_tmp[k]))
                    prev_utility -= p_l * np.sum(np.multiply(l_tmp[k], np.sum(l_tmp, axis=0)+np.sum(load_passive, axis=0)))  ###### 일반적으론 패시브유저도 고려해야함!!
                    prev_utility -= p_soh * np.sum(np.multiply(x_s_tmp[k], np.sum(x_s_tmp, axis=0)))
                    prev_utility -= p_soh * np.sum(np.multiply(x_b_tmp[k], np.sum(x_b_tmp, axis=0)))
                #new_utility += cp.sum(cp.multiply(p_s, x_s[k]) - cp.multiply(p_b, x_b[k]))
                #new_utility -= p_l * cp.sum(cp.power(l[k], 2) + cp.multiply(l[k], np.sum(l_tmp, axis=0) - l_tmp[k] + np.sum(
                #    load_passive, axis=0)))  ###### 일반적으론 패시브유저도 고려해야함!!

                #new_utility -= p_soh * cp.sum(
                #    cp.power(x_s[k], 2) + cp.multiply(x_s[k], np.sum(x_s_tmp, axis=0) - x_s_tmp[k]))
                #new_utility -= p_soh * cp.sum(
                #    cp.power(x_b[k], 2) + cp.multiply(x_b[k], np.sum(x_b_tmp, axis=0) - x_b_tmp[k]))
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
            #difference = gamma * cp.sum(cp.power(cp.vstack([x_s, x_b]) - np.vstack([x_s_tmp, x_b_tmp]), 2)) / 2
            difference = gamma * cp.sum(cp.power(cp.vstack([x_s_single, x_b_single]) - np.vstack([x_s_tmp[i], x_b_tmp[i]]), 2)) / 2
            utility = new_utility - prev_utility - difference
            constraints = []
            constraints += [l_single >= 0]
            constraints += [x_s_single >= 0]
            constraints += [x_b_single >= 0]
            ess_matrix = np.fromfunction(np.vectorize(lambda a, b: 0 if a < b else np.power(alpha, a - b)),
                                         (time, time), dtype=float)

            q_ess = q_min * np.fromfunction(np.vectorize(lambda a, b: np.power(alpha, a - b + 1)), (time, act_user),
                                                  dtype=float) \
                          + beta_s * ess_matrix @ (np.sum(x_s_tmp, axis=0)-x_s_tmp[i]+x_s_single).T - beta_b * ess_matrix @ (np.sum(x_b_tmp, axis=0)-x_b_tmp[i]+x_b_single).T
            constraints += [q_min <= q_ess, q_ess <= q_max]
            constraints += [np.sum(x_s_tmp, axis=0)-x_s_tmp[i]+x_s_single <= c_max]
            constraints += [np.sum(x_b_tmp, axis=0)-x_b_tmp[i]+x_b_single <= c_min]
            prob = cp.Problem(cp.Maximize(utility), constraints)

            result = prob.solve(solver='ECOS')
            #print("total_get user,", i+1, ":", l_single.value-x_s_single.value+x_b_single.value)
            #print("grid load,", i+1, ":", l_single.value)
            #print("ess sell,", i + 1, ":", x_s_single.value)
            #print("ess buy,", i + 1, ":", x_b_single.value)
            #print("ess state :", q_ess.value)

            #if i%100 == 0:
            #    print(i)
                #print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
                #print("### ni ###")
                #print(prev_utility)
                #print(utility.value)
                #print(difference.value)
                #print("x_s :", x_s.value.reshape(act_user, -1))
                #print("x_b :", x_b.value.reshape(act_user, -1))
                #print(i, "l   :", l.value[0])#.reshape(act_user, -1))
            #print(result)
            if new_utility.value-prev_utility-difference.value>0:
                #print("haha")
                #print(l_single.value)
                x_s_tmp[i] = x_s_single.value
                x_b_tmp[i] = x_b_single.value
                l_tmp[i] = l_single.value
                diff += difference.value
                #print("should be same", x_s_tmp[i], x_s_single.value)
                #print("should be same", x_b_tmp[i], x_b_single.value)
                #print("should be same", l_tmp[i], l_single.value)
                #print(l_tmp)
            else:
                #print("nono", result)
                continue

            #x_s_tmp = x_s.value
            #x_b_tmp = x_b.value
            #l_tmp = l.value
            #print("user ", i, "prev util :", -prev_utility, "new util :", -new_utility.value+difference.value, "is cost lower :,", -prev_utility>-new_utility.value+difference.value)
        if iter % 20 == 0:
            print("diff  :", diff)
        #print("EC   :", get_ec(act_user, time, load_matrix, operator_action, np.array([x_s_tmp, x_b_tmp, l_tmp])), "diff   :", diff)
        #if iter %20 == 0:
            #plt.plot(np.sum(load_active, axis=0), color='k')
            #plt.plot(np.sum(l_tmp, axis=0), color='r')

        if diff < 1e-8:
            #plt.show()
            if start is not None:
                a_o = operator_action
                a_f = np.array([x_s_tmp, x_b_tmp, l_tmp])
                print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
                cur_time = timer.time() - start
                data += [[cur_time, a_o, a_f, False]]
                np.save("share_ni_time_tmp.npy", data)
            break

        if start is not None:
            a_o = operator_action
            a_f = np.array([x_s_tmp, x_b_tmp, l_tmp])
            print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
            cur_time = timer.time() - start
            data += [[cur_time, a_o, a_f, False]]
            np.save("share_ni_time_tmp.npy", data)
            #print(l_tmp[0])
        iter += 1
    return result, x_s_tmp, x_b_tmp, l_tmp, data


def step_size_ni_time(act_user, time, load_matrix, operator_action, user_action, d, r):

    update_coef = 0.1
    s = 0.1
    for i in range(2000):
        next_operator_action = operator_action + s * r
        result, x_s, x_b, l, taken_time = follower_action_ni_time(act_user, time, next_operator_action, load_matrix)
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
            if s <= 1e-6:
                break
    return 0, operator_action, user_action


def iterations_ni_time(act_user, time, load_matrix, operator_action, data=[], start=timer.time(), init=None):

    if act_user == 0:
        return None, None, np.array([None, None, 0, True])
    a_o = operator_action
    print("EC   :", get_ec(act_user, time, load_matrix, a_o, np.array([np.zeros((act_user, time)), np.zeros((act_user, time)), load_matrix[:act_user,:]])))
    result, x_s, x_b, l, data = follower_action_ni_time(act_user, time, a_o, load_matrix, start, data, init, gamma=1000)

    a_f = np.array([x_s, x_b, l])

    min_step_size = 1e-6

    print("EC   :", get_ec(act_user, time, load_matrix, a_o, a_f))
    print("PAR  :", get_par(act_user, time, load_matrix, a_o, a_f))

    for i in range(max_iter):
        cur_time = timer.time() - start
        data += [[cur_time, a_o, a_f, True]]
        np.save("share_ni_time_tmp.npy", data)

        print("ITER :", i)
        print("DIRECTION")
        result, d, r, v, g = direction_finding(act_user, time, load_matrix, a_o, a_f)
        print("d :", result)
        #print("r :", r)
        print("STEP")

        b, next_a_o, next_a_f = step_size_ni_time(act_user, time, load_matrix, a_o, a_f, d, r)
        if b != 0:
            print("s :", b)
            a_o = next_a_o
            #_, x_s, x_b, l, _ = follower_action(act_user, time, a_o, load_matrix)
            a_f = next_a_f #np.array([x_s, x_b, l])
        else:
            print("NO UPDATE")
            return a_o, a_f, data

        print("EC  :", get_ec(act_user, time, load_matrix, a_o, a_f))
        print("PAR :", get_par(act_user, time, load_matrix, a_o, a_f))
        if b <= min_step_size:
            break

    return a_o, a_f, data
