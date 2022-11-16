import numpy as np
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pygem import FFD
from pygem.ffd import FFD2D
from geomdl import fitting
from geomdl.visualization import VisMPL as vis

def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T

inputfile = "D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/profile_single_section_2d.curve"
mesh = pd.read_table(inputfile, skiprows=1, delim_whitespace=True, warn_bad_lines=True, error_bad_lines=False, names=['x', 'y', 'z'])

mesh = mesh.to_numpy()   
mesh = mesh[:, 1:] 
angle = 145/180.0 * np.pi
cos = np.cos(angle)
sin = np.sin(angle)
rotation = np.array([[cos, -sin],[sin, cos]])
mesh = np.transpose(rotation.dot(np.transpose(mesh)))
mesh[:, 0] += 0.11
mesh[:, 1] += 0.015

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
# ax.scatter(*mesh.T)
# # ax.axis('equal')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_xlim(0,0.11)
# ax.set_ylim(-0.01,0.01)

# plt.show()

ffd = FFD2D()
ffd.read_parameters("D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/profile_single_section_2d.prm")
print(ffd)

print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

# ffd.array_mu_x[1, 1] = 0.25
ffd.array_mu_y[1, 1] = 0.25
print()
print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

new_mesh = ffd(mesh)
print(type(new_mesh), new_mesh.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(*new_mesh.T, c='green', label='deformed')
ax.scatter(*mesh.T, label='original', c='blue')
ax.scatter(*ffd.control_points().T, s=50, c='red')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.legend()
plt.show()
