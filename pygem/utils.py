"""
Utilities for the affine transformations of the bounding box of the Free Form
Deformation.
"""
import math
from functools import reduce
import numpy as np
from scipy import special
import pybobyqa

def angles2matrix(rot_z=0, rot_y=0, rot_x=0):
    """
    This method returns the rotation matrix for given rotations around z, y and
    x axes.  The output rotation matrix is equal to the composition of the
    individual rotations.  Rotations are counter-clockwise. The default value of
    the three rotations is zero.

    :param float rot_z: rotation angle (in radians) around z-axis.
    :param float rot_y: rotation angle (in radians) around y-axis.
    :param float rot_x: rotation angle (in radians) around x-axis.

    :return: rot_matrix: rotation matrix for the given angles. The matrix shape
        is always (3, 3).
    :rtype: numpy.ndarray

    :Example:

    >>> import pygem.affine as at
    >>> import numpy as np
    >>> from math import radians
    >>> # Example of a rotation around x, y, z axis
    >>> rotz = radians(10)
    >>> roty = radians(20)
    >>> rotx = radians(30)
    >>> rot_matrix = at.angles2matrix(rotz, roty, rotx)

    .. note::

        - The direction of rotation is given by the right-hand rule.
        - When applying the rotation to a vector, the vector should be column
            vector to the right of the rotation matrix.
    """
    rot_matrix = []
    if rot_z:
        cos = math.cos(rot_z)
        sin = math.sin(rot_z)
        rot_matrix.append(
            np.array([cos, -sin, 0, sin, cos, 0, 0, 0, 1]).reshape((3, 3)))
    if rot_y:
        cos = math.cos(rot_y)
        sin = math.sin(rot_y)
        rot_matrix.append(
            np.array([cos, 0, sin, 0, 1, 0, -sin, 0, cos]).reshape((3, 3)))
    if rot_x:
        cos = math.cos(rot_x)
        sin = math.sin(rot_x)
        rot_matrix.append(
            np.array([1, 0, 0, 0, cos, -sin, 0, sin, cos]).reshape((3, 3)))
    if rot_matrix:
        return reduce(np.dot, rot_matrix[::-1])
    return np.eye(3)

def angles2matrix2d(theta=0):
    rot_matrix = []
    if theta:
        cos = math.cos(theta)
        sin = math.sin(theta)
        rot_matrix = np.array([cos, -sin, sin, cos]).reshape((2, 2))
        return rot_matrix
    return np.eye(2)      

def fit_affine_transformation(points_start, points_end):
    """
    Fit an affine transformation from starting points to ending points through a
    least square procedure.

    :param numpy.ndarray points_start: set of starting points.
    :param numpy.ndarray points_end: set of ending points.

    :return: transform_vector: function that transforms a vector according to
        the affine map. It takes a source vector and return a vector transformed
        by the reduced row echelon form of the map.
    :rtype: function

    :Example:

    >>> import pygem.affine as at

    >>> # Example of a rotation (affine transformation)
    >>> p_start = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]])
    >>> p_end = np.array([[0,1,0], [-1,0,0], [0,0,1], [0,0,0]])
    >>> v_test = np.array([1., 2., 3.])
    >>> transformation = at.affine_points_fit(p_start, p_end)
    >>> v_trans = transformation(v_test)
    """
    if len(points_start) != len(points_end):
        raise RuntimeError("points_start and points_end must be of same size.")

    dim = len(points_start[0])
    if len(points_start) < dim:
        raise RuntimeError(
            "Too few starting points => under-determined system.")

    def pad_column_ones(x):
        """ Add right column of 1.0 to the given 2D numpy array """
        return np.hstack([x, np.ones((x.shape[0], 1))])

    def unpad_column(x):
        """ Remove last column to the given 2D numpy array """
        return x[:, :-1]

    def transform(src):

        shape = src.shape

        X = pad_column_ones(points_start)
        Y = pad_column_ones(points_end)

        A, res, rank, _ = np.linalg.lstsq(X, Y, rcond=None)
        # TODO add check condition number
        #if np.linalg.cond(A) >= 1 / sys.float_info.epsilon:
        #    raise RuntimeError(
        #            "Error: singular matrix. Points are probably coplanar.")
        return unpad_column(
                np.dot(
                    pad_column_ones(np.atleast_2d(src)),
                    A)
                ).reshape(shape)

    return transform


def phy2ffd_map2d(ctrl_pts, phy_coords, dim_x, dim_y):
    '''
    to be filled
    '''
    assert ctrl_pts.shape[0] == dim_x * dim_y, "shoule have {} control points, {} given.".format(dim_x*dim_y, ctrl_pts.shape[0])
    phy_coord = np.array([0, 0])

    def obj_func_x(x):
        u, v = x[0], x[1]
        # phy_coord = np.array([x[2], x[3]])

        bernstein_x = np.zeros((dim_x, 1))
        bernstein_y = np.zeros((dim_y, 1))

        for i in range(0, dim_x):
            aux1 = np.power((1 - u), dim_x - 1 - i)
            aux2 = np.power(u, i)
            bernstein_x[i, :] = (special.binom(dim_x - 1, i) *
                                    np.multiply(aux1, aux2))

        for i in range(0, dim_y):
            aux1 = np.power((1 - v), dim_y - 1 - i)
            aux2 = np.power(v, i)
            bernstein_y[i, :] = special.binom(dim_y - 1,
                                                i) * np.multiply(aux1, aux2)

        # for i in range(0, dim_t_mu):
        #     aux1 = np.power((1 - points[:, 2]), dim_t_mu - 1 - i)
        #     aux2 = np.power(points[:, 2], i)
        #     bernstein_z[i, :] = special.binom(dim_t_mu - 1,
        #                                       i) * np.multiply(aux1, aux2)

        aux_x = 0.
        # aux_y = 0.
        # aux_z = 0.

        for j in range(0, dim_y):
            # for k in range(0, dim_t_mu):
            #     bernstein_yz = np.multiply(bernstein_y[j, :],
            #                                bernstein_z[k, :])
            for i in range(0, dim_x):
                aux = np.multiply(bernstein_x[i, :], bernstein_y[j, :])
                aux_x += aux * ctrl_pts[dim_y*i + j][0]
                # aux_y += aux * ctrl_pts[dim_y*i + j][1]

        # print(aux_x)
        # print(aux_y)
        
        return (aux_x - phy_coord[0])**2

    def obj_func_y(x):
        u, v = x[0], x[1]
        # phy_coord = np.array([x[2], x[3]])

        bernstein_x = np.zeros((dim_x, 1))
        bernstein_y = np.zeros((dim_y, 1))

        for i in range(0, dim_x):
            aux1 = np.power((1 - u), dim_x - 1 - i)
            aux2 = np.power(u, i)
            bernstein_x[i, :] = (special.binom(dim_x - 1, i) *
                                    np.multiply(aux1, aux2))

        for i in range(0, dim_y):
            aux1 = np.power((1 - v), dim_y - 1 - i)
            aux2 = np.power(v, i)
            bernstein_y[i, :] = special.binom(dim_y - 1,
                                                i) * np.multiply(aux1, aux2)

        # for i in range(0, dim_t_mu):
        #     aux1 = np.power((1 - points[:, 2]), dim_t_mu - 1 - i)
        #     aux2 = np.power(points[:, 2], i)
        #     bernstein_z[i, :] = special.binom(dim_t_mu - 1,
        #                                       i) * np.multiply(aux1, aux2)

        # aux_x = 0.
        aux_y = 0.
        # aux_z = 0.

        for j in range(0, dim_y):
            # for k in range(0, dim_t_mu):
            #     bernstein_yz = np.multiply(bernstein_y[j, :],
            #                                bernstein_z[k, :])
            for i in range(0, dim_x):
                aux = np.multiply(bernstein_x[i, :], bernstein_y[j, :])
                # aux_x += aux * ctrl_pts[dim_y*i + j][0]
                aux_y += aux * ctrl_pts[dim_y*i + j][1]

        # print(aux_x)
        # print(aux_y)
        
        return (aux_y - phy_coord[1])**2

    uv_all = np.zeros_like(phy_coords)


    for i in range(phy_coords.shape[0]):
        phy_coord = phy_coords[i, :]

        lower = np.array([0.0, 0.0])
        upper = np.array([1.0, 1.0])

        soln = pybobyqa.solve(objfun=obj_func_x, x0=uv_all[i-1, :])#, bounds=(lower, upper)
        uv_all[i, 0] = soln.x[0]
        soln = pybobyqa.solve(objfun=obj_func_y, x0=uv_all[i-1, :])#, bounds=(lower, upper)
        uv_all[i, 1] = soln.x[1]
        
    
    return uv_all

def ffd2phy_map2d(ctrl_pts, uv_coords, dim_x, dim_y):
    uv_coords = np.transpose(uv_coords)
    u, v = uv_coords[0, :], uv_coords[1, :]
    # phy_coord = np.array([x[2], x[3]])
    n_rows = uv_coords.shape[1]

    bernstein_x = np.zeros((dim_x, n_rows))
    bernstein_y = np.zeros((dim_y, n_rows))

    for i in range(0, dim_x):
        aux1 = np.power((1 - u), dim_x - 1 - i)
        aux2 = np.power(u, i)
        bernstein_x[i, :] = (special.binom(dim_x - 1, i) *
                                np.multiply(aux1, aux2))

    for i in range(0, dim_y):
        aux1 = np.power((1 - v), dim_y - 1 - i)
        aux2 = np.power(v, i)
        bernstein_y[i, :] = special.binom(dim_y - 1,
                                            i) * np.multiply(aux1, aux2)

    # for i in range(0, dim_t_mu):
    #     aux1 = np.power((1 - points[:, 2]), dim_t_mu - 1 - i)
    #     aux2 = np.power(points[:, 2], i)
    #     bernstein_z[i, :] = special.binom(dim_t_mu - 1,
    #                                       i) * np.multiply(aux1, aux2)

    aux_x = 0.
    aux_y = 0.
    # aux_z = 0.

    for j in range(0, dim_y):
        # for k in range(0, dim_t_mu):
        #     bernstein_yz = np.multiply(bernstein_y[j, :],
        #                                bernstein_z[k, :])
        for i in range(0, dim_x):
            aux = np.multiply(bernstein_x[i, :], bernstein_y[j, :])
            aux_x += aux * ctrl_pts[dim_y*i + j][0]
            aux_y += aux * ctrl_pts[dim_y*i + j][1]

    phy_coords = np.transpose(np.vstack((aux_x, aux_y)))
    return phy_coords