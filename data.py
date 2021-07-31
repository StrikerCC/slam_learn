#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   data.py    
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/29/2021 2:13 PM   Cheng Chen    1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt


def reorder_data(points):
    return np.sort(points, axis=0)


def read_data(reorder=False, show=False):
    dimension = 3
    sperse = 10
    file_fix, file_move = r'./data/bunny.txt', r'./data/bunny-Copy.txt'
    array_fix_org, array_move_org = np.loadtxt(file_fix), np.loadtxt(file_move)
    array_fix, array_move = array_fix_org[::sperse], array_move_org[::sperse]
    array_move_reorder = array_move if not reorder else reorder_data(array_move)

    """vis"""
    if show:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(array_fix[:, 0], array_fix[:, 1], array_fix[:, 2])
        ax.scatter(array_move[:, 0], array_move[:, 1], array_move[:, 2])
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        plt.show()

    print('read ', len(array_fix_org), 'points from txt, using ',  len(array_fix), 'points for registration')
    print('x range ', np.min(array_fix[:, 0]), ' ', np.max(array_fix[:, 0]))
    print('y range ', np.min(array_fix[:, 1]), ' ', np.max(array_fix[:, 1]))
    print('z range ', np.min(array_fix[:, 2]), ' ', np.max(array_fix[:, 2]))
    print('two point sets are', np.mean(np.linalg.norm(array_fix-array_move, axis=1), axis=0), 'away from each other')
    return array_fix, array_move, array_move_reorder


def main():
    read_data(show=True)


if __name__ == '__main__':
    main()