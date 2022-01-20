import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function, simulate_policy

from Problem_1.utils import visualize_value_function_trajectory

# from Problem_1.utils import simulate_policy


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2
    terminal_mask = tf.cast(terminal_mask, tf.bool)
    # perform value iteration
    for _ in range(1000): #changed to i for debugging (and needs to be 1000 iters)
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        pxp = tf.zeros([sdim, sdim])
        possible_values = np.zeros((sdim, adim))
        V_new = np.zeros(sdim)
        V_prev = tf.cast(V, tf.float32)

        for u in range(adim):
            pxp = tf.convert_to_tensor(Ts[u], dtype=tf.float32) 
            temp0 = reward[:,u] + gam*tf.linalg.matvec(pxp, V_prev)
            possible_values[:,u] = tf.where(terminal_mask, reward[:,u], temp0)
        V_new = tf.convert_to_tensor(np.max(possible_values, axis=1), dtype=tf.float32)
        policy = np.argmax(possible_values, axis=1)
        V = V_new
        err = tf.norm(V_new - V_prev)
       
        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V, policy


# value iteration ##############################################################
def main():
    # generate the problem
    problem = generate_problem()
    n = problem["n"]
    sdim, adim = n * n, 1

    # create the terminal mask vector
    terminal_mask = np.zeros([sdim])
    terminal_mask[problem["pos2idx"][19, 9]] = 1.0
    terminal_mask = tf.convert_to_tensor(terminal_mask, dtype=tf.float32)

    # generate the reward vector
    reward = np.zeros([sdim, 4])
    reward[problem["pos2idx"][19, 9], :] = 1.0
    reward = tf.convert_to_tensor(reward, dtype=tf.float32)

    gam = 0.95
    V_opt, p_opt = value_iteration(problem, reward, terminal_mask, gam)
    trajectory = simulate_policy(problem, p_opt)
    traj_coords = problem["idx2pos"][trajectory]
    print(traj_coords)
    plt.figure(213)
    visualize_value_function_trajectory(np.array(V_opt).reshape((n, n)), traj_coords)
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()
