import math, pdb, os, sys

import numpy as np, tensorflow as tf, matplotlib.pyplot as plt, matplotlib.colors as colors
import pickle

infile = open('value_iteration_policy','rb')
VI_data = pickle.load(infile)
infile.close()

infile = open('Q_learning_policy','rb')
Q_data = pickle.load(infile)
infile.close()

u_VI = VI_data["u"]
v_VI = VI_data["v"]
u_Q = Q_data["u"]
v_Q = Q_data["v"]
m, n = np.shape(u_VI)
agreement = np.zeros((m,n))
U = u_VI==u_Q
V = v_VI==v_Q
# agreement = ((u_VI==u_Q) == (v_VI==v_Q)).astype(int)
for i in range(m):
    for j in range(n):
        if U[i,j] and V[i,j]:
            agreement[i,j] = 1

cmap = colors.ListedColormap(['black', 'white'])

print(agreement)
plt.figure(201)
plt.imshow(agreement, origin="lower", cmap=cmap)
plt.title("policy agreement")
plt.show()
