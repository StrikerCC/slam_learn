# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/19/21 6:55 PM
"""
import numpy as np
import transforms3d


def kpconv_kernel_points(k, radius):
    """"""
    '''init each kernel point location'''
    points = np.ones(3, k)
    angle = np.arange(0, 360, 360/k)
    points[:, 0] = np.cos(angle)
    points[:, 1] = np.sin(angle)

    '''start moving'''
    for i in range(1000):
        '''compute energy'''
        '''compute Jacobin and hessian'''
        '''gradient descent'''

    return points


def energy(points_):
    """add forces"""
    center = np.mean(points_, axis=-1)
    energy_rep = 1 / np.linalg.norm(points_-center, axis=-1)
    # energy_rep = np.sum(energy_rep, axis=0)
    energy_att = np.linalg.norm(points_, axis=-1)
    energy_ = np.sum(energy_att + energy_rep, axis=0)
    energy_prime = np.sum()
    return energy_,
