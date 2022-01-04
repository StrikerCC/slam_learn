# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/11/21 10:50 AM
"""

import numpy as np
import matplotlib.pyplot as plt


def quadratic(a, b, c, x):
    shape_x = x.shape if isinstance(x, np.ndarray) else (1,)
    if len(shape_x) > 1:
        x = x.flatten()
    A = np.array([a, b, c])
    x_ = np.array([pow(x, 2), x, np.ones_like(x)])
    x_prime = np.array([2 * x, np.ones_like(x), np.zeros_like(x)])
    x_prime_prime = np.array([2 * np.ones_like(x), np.zeros_like(x), np.zeros_like(x)])
    return np.dot(A, x_).reshape(shape_x), np.dot(A, x_prime).reshape(shape_x), np.dot(A, x_prime_prime).reshape(shape_x)


def quadratic_2(a1, b1, c1, x, a2, b2, c2, y):
    z1, z1_prime, z1_prime_prime = quadratic(a1, b1, c1, x)
    z2, z2_prime, z2_prime_prime = quadratic(a2, b2, c2, y)
    z1_prime_prime_z2_prime_prime = np.zeros((len(z1), 2, 2))
    z1_prime_prime_z2_prime_prime[:, 0, 0] = z1_prime_prime
    z1_prime_prime_z2_prime_prime[:, 1, 1] = z2_prime_prime
    return z1 + z2, np.array([z1_prime, z2_prime]), z1_prime_prime_z2_prime_prime


def tuplti(a, b, c, d, x):
    A = np.asarray([a, b, c, d])
    x_ = np.asarray([pow(x, 3), pow(x, 2), x, np.ones_like(x)])
    x_prime = np.asarray([3 * pow(x, 2), 2 * x, np.ones_like(x), np.zeros_like(x)])
    x_prime_prime = np.asarray([6 * x, 2 * np.ones_like(x), np.zeros_like(x), np.zeros_like(x)])
    x_prime_prime_prime = np.asarray([6 * np.ones_like(x), np.zeros_like(x), np.zeros_like(x), np.zeros_like(x)])
    return np.dot(A, x_), np.dot(A, x_prime), np.dot(A, x_prime_prime), np.dot(A, x_prime_prime_prime)
