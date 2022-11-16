import numpy as np

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pygem.utils import phy2ffd_map2d

if __name__ == '__main__':
    ctrl_points = np.array([
        [0, 0],
        [0, 0.5],
        [1, 0],
        [1, 1],
        [2, 0.5],
        [2, 1]
    ])

    phy_coords = np.array([
        [1.0, 0.5],
        [1.5, 0.5]
    ])

    dim_x, dim_y = 3, 2
    uv = phy2ffd_map2d(ctrl_pts=ctrl_points, phy_coords=phy_coords, dim_x=dim_x, dim_y=dim_y)
    print(uv)