#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    :   utils.py
@Contact :   chengc0611@gmail.com
@License :   

@Modify Time      @Author       @Version    @Desciption
------------      -------       --------    -----------
7/22/2021 2:17 PM   Cheng CHen    1.0         None
'''

# import lib
import numpy as np
import matplotlib.pyplot as plt

def linear_least_square(A, b):
    """
    solve the a the combination of columns in A to be projection of b on A
    :param A: space
    :param b: desired projection
    :return:
    """
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    assert A.shape[0] > A.shape[1], A.shape
    assert A.shape[0] == b.shape[0], str(A.shape) + ', ' + str(b.shape)
    return np.matmul(np.linalg.inv(np.matmul(A.T, A)), np.matmul(A.T, b))


def ordinary_least_square(A, b):

    return


def main():
    # make data to fit
    paras = [3.4, -1.5, -2.2, 3.1, 2.1, 1.1]                                       # parameters
    # a, b = 0.1, 0.2                                       # parameters

    t = np.linspace(-100, 100, 100)                                          # time

    for i in range(len(paras)):
        A = np.ones(t.shape) if i == 0 else np.vstack((A, np.power(t, i)))   # time square, time, constant
    A = A.T

    amp = 1
    noise = np.random.random(A.shape)
    A_noise = A + noise * amp

    # para_expect = np.array([a, b, c]).reshape(A.shape[1], 1)
    para_expect = np.array(paras).reshape(A.shape[1], 1)
    proj = np.matmul(A_noise, para_expect)                                              # expect observations

    plt.scatter(t, proj)
    # proj = np.sort(proj, axis=0)
    # plt.scatter(t, proj)
    plt.show()

    # solve linear_least_square
    x_ = linear_least_square(A, proj)
    proj_ = np.matmul(A, x_)                                            # observation according to line fitting
    # y_ = np.squeeze(y_, axis=-1).reshape(t.shape[0], -1)
    print(x_)
    plt.scatter(t, proj)
    # plt.show()
    plt.plot(t, proj_)
    plt.show()


if __name__ == "__main__":
    main()
