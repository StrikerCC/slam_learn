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
import open3d as o3

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


def read_data_(reorder=False, show=False):
    file_fix = r'./data/point_cloud.txt'
    array_fix = np.loadtxt(file_fix)
    # array_fix = array_fix[:10000]
    print(array_fix.shape)
    pc = o3.geometry.PointCloud()
    pc.points = o3.utility.Vector3dVector(array_fix)
    pc.voxel_down_sample(1)
    o3.visualization.draw_geometries([pc])


def main():
    read_data_(show=True)


if __name__ == '__main__':
    main()