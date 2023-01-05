import numpy as np
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pygem import FFD
from geomdl import fitting
from geomdl.visualization import VisMPL as vis

def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T

# mesh = mesh_points()
inputfile = "D:/BaiduSyncdisk/graduate/parametrization/FFD/PyGeM/tests/test_datasets/profile.curve"
mesh = pd.read_table(inputfile, skiprows=1, delim_whitespace=True, warn_bad_lines=True, error_bad_lines=False, names=['x', 'y', 'z'])

# mesh = mesh.sort_values(by=['x','y','z'])
mesh = mesh.to_numpy()                  
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*mesh.T)
# ax.axis('auto')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()

ffd = FFD()
ffd.read_parameters("D:/BaiduSyncdisk/graduate/parametrization/FFD/PyGeM/tests/test_datasets/profile.prm")
print(ffd)

print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

# ffd.array_mu_x[1, 1, 1] = 0.8
ffd.array_mu_z[1, 1, 1] = 2
ffd.array_mu_z[1, 2, 2] = 2
ffd.array_mu_z[1, 2, 2] = 2
ffd.array_mu_z[2, 2, 2] = 2

print()
print('Movements of point[{}, {}, {}] along x: {}'.format(1, 1, 1, ffd.array_mu_x[1, 1, 1]))
print('Movements of point[{}, {}, {}] along z: {}'.format(1, 1, 1, ffd.array_mu_z[1, 1, 1]))

new_mesh = ffd(mesh)
print(type(new_mesh), new_mesh.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(*new_mesh.T, c='green', label='deformed')
ax.scatter(*mesh.T, label='original', c='blue')
ax.scatter(*ffd.control_points().T, s=50, c='red')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
ax.legend()
plt.show()
