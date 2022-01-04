# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 10/13/21 7:18 PM
"""
import numpy as np
import matplotlib.pyplot as plt
from functions import quadratic, quadratic_2, tuplti


def approximation(func):
    x = np.arange(-2, 2.1, 0.1)
    y, y_prime, y_prime_prime, y_prime_prime_prime = func(-0.5, 0.5, 1, 0, x)
    # y, y_prime, y_prime_prime = func(0.5, 0, 0, x)

    for x_root in x:
        # x_root = -0.5
        y_root, y_prime_root, y_prime_prime_root, y_prime_prime_prime_root = func(-0.5, 0.5, 1, 0, x_root)
        # y_root, y_prime_root, y_prime_prime_root = func(0.5, 0, 0, x_root)

        # x_delta = x-x_root
        x_delta = np.arange(-1, 1.1, 0.1)
        x_ = x_delta + x_root

        A_ = np.array([y_root, y_prime_root, y_prime_prime_root])
        X_ = np.array([np.ones_like(x_delta), x_delta, pow(x_delta, 2)])
        y_ = np.dot(A_, X_)
        # A_ = np.array([y_root, y_prime_root, y_prime_prime_root, y_prime_prime_prime_root])
        # X = np.array([np.ones_like(x_delta), x_delta, pow(x_delta, 2), pow(x_delta, 3)])

        # y_ = quadratic()
        plt.plot(x, y, x_, y_)
        plt.show()


def netwon(func):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # plt.ion()

    x = np.arange(-10, 10.1, 0.1)
    y = np.arange(-10, 10.1, 0.1)
    x, y = np.meshgrid(x, y)
    x_shape = x.shape
    x, y = x.flatten(), y.flatten()
    z, z_prime, z_prime_prime = func(0.5, 0, 0, x, 0.5, 0, 0, y)
    x, y, z = x.reshape(x_shape), y.reshape(x_shape), z.reshape(x_shape)

    x_root, y_root = 5, 10
    for i in range(100):
        # z_root, z_prime_root, z_prime_prime_root, y_prime_prime_prime_root = func(-0.5, 0.5, 1, 0, x_root)
        z_root, z_prime_root, z_prime_prime_root = func(0.5, 0, 0, x_root, 0.5, 0, 0, y_root)
        z_prime_prime_root = z_prime_prime_root.squeeze()
        # print(z_prime_root)
        # print(z_prime_prime_root)
        # print(np.dot(np.linalg.inv(z_prime_prime_root), z_prime_root))
        x_step_y_step = - np.dot(np.linalg.inv(z_prime_prime_root), z_prime_root)
        x_step = x_step_y_step[0]
        y_step = x_step_y_step[1]
        x_step *= 0.5
        y_step *= 0.5

        print('x', x_root)
        print('y', y_root)

        '''plot'''
        x_range = np.arange(-1, 1.1, 0.1)
        y_range = np.arange(-1, 1.1, 0.1)
        x_range += x_root
        y_range += y_root

        x_, y_ = np.meshgrid(x_range, y_range)
        x__shape = x_.shape
        x_, y_ = x_.flatten(), y_.flatten()
        z_, *_ = func(0.5, 0, 0, x_, 0.5, 0, 0, y_)
        x_, y_, z_ = x_.reshape(x__shape), y_.reshape(x__shape), z_.reshape(x__shape)

        '''plot'''
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.plot_wireframe(x_, y_, z_, colors='g')
        ax.plot_wireframe(x, y, z)
        plt.show()
        # plt.ion()
        # plt.pause(1)
        # plt.close()

        # if x_step < 0.01 and y_step < 0.01:
        #     break
        x_root += x_step
        y_root += y_step


def gassian_netwon(func):
    x = np.arange(-2, 2.1, 0.1)
    y, y_prime, y_prime_prime, y_prime_prime_prime = func(-0.5, 0.5, 1, 0, x)

    x_root = 0
    for i in range(100):
        y_root, y_prime_root, y_prime_prime_root, y_prime_prime_prime_root = func(-0.5, 0.5, 1, 0, x_root)
        x_step = - y_prime_root
        x_step *= 1

        print('x', x_root)
        print('y', y_root)

        '''plot'''
        x_range = np.arange(-1, 1.1, 0.1)
        x_ = x_range + x_root

        A_ = np.array([y_root, y_prime_root, y_prime_prime_root])
        X_ = np.array([np.ones_like(x_range), x_range, pow(x_range, 2)])
        y_ = np.dot(A_, X_)
        plt.plot(x, y, x_, y_)
        plt.show()

        x_root += x_step


def gradient_descent(func):
    x = np.arange(-2, 2.1, 0.1)
    y, y_prime, y_prime_prime, y_prime_prime_prime = func(-0.5, 0.5, 1, 0, x)

    x_root = 0
    for i in range(100):
        y_root, y_prime_root, y_prime_prime_root, y_prime_prime_prime_root = func(-0.5, 0.5, 1, 0, x_root)
        x_step = - y_prime_root / 0.01
        x_step *= 1

        print('x', x_root)
        print('y', y_root)

        '''plot'''
        x_range = np.arange(-1, 1.1, 0.1)
        x_ = x_range + x_root

        A_ = np.array([y_root, y_prime_root, y_prime_prime_root])
        X_ = np.array([np.ones_like(x_range), x_range, pow(x_range, 2)])
        y_ = np.dot(A_, X_)
        plt.plot(x, y, x_, y_)
        plt.show()

        x_root += x_step


def main():
    netwon(quadratic_2)
    # approximation(tuplti)
    x = np.arange(-2, 2.1, 0.01)
    y = quadratic(1, 0, 0, x)
    # print(y)
    plt.plot(x, y[0])
    # plt.show()

    y = tuplti(-0.5, 0.5, 1, 0, x)
    plt.plot(x, y[0])
    # plt.show()


if __name__ == '__main__':
    main()
