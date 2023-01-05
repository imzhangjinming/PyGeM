import numpy as np
import mpl_toolkits.mplot3d
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
import pandas as pd

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from pygem import FFD
from pygem.ffd import FFD2D, FFD2D_irregular

def mesh_points(num_pts=2000):
    indices = np.arange(0, num_pts, dtype=float) + 0.5

    phi = np.arccos(1 - 2*indices/num_pts)
    theta = np.pi * (1 + 5**0.5) * indices

    return np.array([np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)]).T

inputfile = "D:/BaiduSyncdisk/graduate/parametrization/FFD/PyGeM/tests/test_datasets/CDA.curve"
mesh = pd.read_table(inputfile, skiprows=1, delim_whitespace=True, warn_bad_lines=True, error_bad_lines=False, names=['x1', 'pressure_side', 'x2', 'suction_side'])

mesh = mesh.to_numpy()   
pressure_side = mesh[:, :2].copy().astype(np.float)
suction_side = mesh[:, 2:].copy().astype(np.float)

# fig = plt.figure(figsize=(8, 8))
# ax = fig.add_subplot(111)
# ax.scatter(*pressure_side.T)
# ax.scatter(*suction_side.T)
# ax.axis('equal')
# ax.set_xlabel('x/mm')
# ax.set_ylabel('y/mm')
# # ax.set_xlim(0,0.11)
# # ax.set_ylim(-0.01,0.01)
# plt.show()

ffd = FFD2D_irregular()
ffd.read_parameters("D:/BaiduSyncdisk/graduate/parametrization/FFD/PyGeM/tests/test_datasets/CDA-ir.prm")
print(ffd)

print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

# ffd.array_mu_x[1, 1] = 0.25
ffd.array_mu_y[1, 1] = 1
print()
print('Movements of point[{}, {}] along x: {}'.format(1, 1, ffd.array_mu_x[1, 1]))
print('Movements of point[{}, {}] along y: {}'.format(1, 1, ffd.array_mu_y[1, 1]))

new_pressure_side = ffd(pressure_side)
new_suction_side = ffd(suction_side)

f = open(r'D:/BaiduSyncdisk/graduate/parametrization/FFD/PyGeM/tests/test_datasets/CDA_deformed-ir.curve', 'w')
f.write('# CDA section, mm\n')
for i in range(new_pressure_side.shape[0]):
    f.write('{} {} {} {}\n'.format(new_pressure_side[i, 0], new_pressure_side[i, 1], new_suction_side[i,0], new_suction_side[i,1]))

f.close()
# print(type(new_mesh), new_mesh.shape)

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

ax.scatter(new_pressure_side[:,0], new_pressure_side[:,1], c='green', label='deformed')
ax.scatter(new_suction_side[:,0], new_suction_side[:,1], c='green')

ax.scatter(pressure_side[:,0], pressure_side[:,1], label='original', c='blue')
ax.scatter(suction_side[:,0], suction_side[:,1], c='blue')

ax.scatter(ffd.control_points()[:,0], ffd.control_points()[:,1], label='control points', s=50, c='red')
ax.set_xlabel('x/mm')
ax.set_ylabel('y/mm')
ax.legend()
ax.axis('equal')
plt.show()
print(1)
