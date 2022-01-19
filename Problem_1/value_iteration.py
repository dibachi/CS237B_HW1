import os, sys, pdb, math, pickle, time

import matplotlib
import tensorflow as tf, numpy as np, matplotlib.pyplot as plt
from tqdm import tqdm

from utils import generate_problem, visualize_value_function


def value_iteration(problem, reward, terminal_mask, gam):
    Ts = problem["Ts"]
    sdim, adim = Ts[0].shape[-1], len(Ts)  # state and action dimension
    V = tf.zeros([sdim])

    assert terminal_mask.ndim == 1 and reward.ndim == 2

    # perform value iteration
    for i in range(10): #changed to i for debugging (and needs to be 1000 iters)
        ######### Your code starts here #########
        # perform the value iteration update
        # V has shape [sdim]; sdim = n * n is the total number of grid state
        # Ts is a 4 element python list of transition matrices for 4 actions

        # reward has shape [sdim, 4] - represents the reward for each state
        # action pair

        # terminal_mask has shape [sdim] and has entries 1 for terminal states

        # compute the next value function estimate for the iteration
        # compute err = tf.linalg.norm(V_new - V_prev) as a breaking condition
        pxp = tf.zeros([sdim])
        possible_values = np.zeros(adim) #np is on purpose for assignment to work
        V_new = np.zeros(sdim) #we assign values in the loop, then convert to tensor for V update
        V_prev = tf.cast(V, tf.float32)
        # policy = np.zeros(sdim)
        
        for x in range(sdim):
            px = np.zeros(sdim) #initial probability distribution over states
            px[x] = 1 #we know the state is x, so px = 1 @ x
            px = tf.convert_to_tensor(px, dtype=tf.float32)
            for u in range(adim):
                pxp = tf.linalg.matvec(tf.convert_to_tensor(Ts[u], dtype=tf.float32), px) #get probability distributions over all x' for each action u
                temp0 = reward[x, u] + gam*tf.tensordot(pxp, V_prev, 1) #bellman update, dot does multiplication and sumation in place
                if terminal_mask[x] == 1: #if terminal state, update is just the reward @ x, u
                    temp0 = reward[x, u] #temp0 is a tensor in both cases
                possible_values[u] = temp0 #save value for action u at current state
            V_new[x] = np.max(possible_values) #write the optimal value at state x for optimal action
            # policy[x] = np.argmax(possible_values)
        V_new = tf.convert_to_tensor(V_new) #for compatibility
        V = V_new #write updated values from temporary variable to V
        err = tf.norm(tf.cast(V_new, tf.float32) - tf.cast(V_prev, tf.float32)) #calc error term
        print(f"Got here {i}")
        ######### Your code ends here ###########

        if err < 1e-7:
            break

    return V 


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
    V_opt = value_iteration(problem, reward, terminal_mask, gam)

    plt.figure(213)
    visualize_value_function(np.array(V_opt).reshape((n, n)))
    plt.title("value iteration")
    plt.show()


if __name__ == "__main__":
    main()
