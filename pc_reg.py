#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   pc_reg.py    
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/31/2021 1:23 PM   Cheng Chen    1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt

from data import read_data
from utils import rms_error

# global variables
dimension = 3


def compute_rigid_transf(pc_target, pc_src):
    assert isinstance(pc_target, np.ndarray) and isinstance(pc_src, np.ndarray)
    assert pc_target.shape == pc_src.shape
    assert pc_target.shape[1] == pc_src.shape[1] == dimension
    # compute H
    H = np.matmul(pc_src, pc_target.T)
    U, S, V_T = np.linalg.svd(H)
    R = np.matmul(V_T.T, U)
    t = pc_target - np.matmul(R, pc_src)

    # assert np.linalg.det(R) == 1
    return R, t


def main():
    array_fix, array_move, array_move_reorder = read_data(show=False)
    """vis"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
    ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    R, t = compute_rigid_transf(pc_target=array_fix, pc_src=array_move)
    # tranf_rigid = np.concatenate(R, t, axis=0)
    array_move = np.dot(R, array_move) + t
    print('rms error is ', rms_error(array_fix, array_move))

    """vis"""
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
    ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
    return


if __name__ == '__main__':
    main()
