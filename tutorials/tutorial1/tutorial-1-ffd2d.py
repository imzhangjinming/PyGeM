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

# mesh = mesh_points()
inputfile = "D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/profile_single_section_2d.curve"
mesh = pd.read_table(inputfile, skiprows=1, delim_whitespace=True, warn_bad_lines=True, error_bad_lines=False, names=['x', 'y', 'z'])

# mesh = mesh.sort_values(by=['x','y','z'])
mesh = mesh.to_numpy()   
mesh = mesh[:, 1:]               
# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
# ax.scatter(*mesh.T)
# ax.axis('auto')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# plt.show()

ffd = FFD2D()
ffd.read_parameters("D:/ZJM/graduate/叶型参数化方法/FFD/PyGeM/tests/test_datasets/profile_single_section_2d.prm")
print(ffd)

print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

ffd.array_mu_x[1, 2] = 0.25
ffd.array_mu_y[1, 2] = 0.25
print()
print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

new_mesh = ffd(mesh)
print(type(new_mesh), new_mesh.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(*new_mesh.T, c='green')
ax.scatter(*ffd.control_points().T, s=50, c='red')

ax.scatter(*mesh.T)
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
plt.show()
