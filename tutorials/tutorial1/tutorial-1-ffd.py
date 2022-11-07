import numpy as np
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pygem import FFD


def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T

# mesh = mesh_points()
inputfile = "D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/rotor67.xyz"
mesh = pd.read_table(inputfile, skiprows=1, delim_whitespace=True,
                         names=['x', 'y', 'z'])
mesh = mesh.to_numpy()                  
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*mesh.T)
# ax.axis('auto')
# plt.show()

ffd = FFD()
ffd.read_parameters("D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/rotor67.prm")
print(ffd)

print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

# ffd.array_mu_x[1, 1, 1] = 0.8
ffd.array_mu_z[1, 1, 1] = 0.8
print()
print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

new_mesh = ffd(mesh)
print(type(new_mesh), new_mesh.shape)


ax = plt.figure(figsize=(8, 8)).add_subplot(111, projection='3d')
ax.scatter(*new_mesh.T)
ax.scatter(*ffd.control_points().T, s=50, c='red')
ax.scatter(*mesh.T, c='black')
plt.show()
