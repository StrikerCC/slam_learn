# -*- coding: utf-8 -*-

"""
@author: Cheng Chen
@email: chengc0611@gmail.com
@time: 8/30/21 10:54 AM
"""

import numpy as np
import matplotlib.pyplot as plt

import utils


def line_fitting(domain, observations, num_power=1, method='least-square'):
    """
    solve this by ordinary_least_square, give a combination of columns in A to be projection of b on A (
    :param num_power:
    :param domain:
    :param observations: desired projection
    :return:
    """
    assert isinstance(observations, np.ndarray)
    assert observations.shape[0] > 2, 'At least 2 observations are required for line-fitting, but got  ' + \
                                      str(observations.shape)
    t = np.linspace(domain[0], domain[1], len(observations))  # define-range
    A = np.zeros((len(observations), num_power))  # observation-power matrix
    for i in range(num_power):
        A[:, i] = np.power(t, i)
    para_exp = None
    if method == 'least-square':
        para_exp = utils.least_square(A, observations)
    elif method == 'ransac':
        para_exp = utils.ransac(A, observations)
    return para_exp


def ridge_regression(A, b, lambda_):
    """
    solve the a the combination of columns in A to be projection of b on A
    :param A: space
    :param b: desired projection
    :return:
    """
    assert isinstance(A, np.ndarray) and isinstance(b, np.ndarray)
    assert A.shape[0] > A.shape[1] and len(A.shape) == 2, A.shape
    assert A.shape[0] == b.shape[0], str(A.shape) + ', ' + str(b.shape)
    return np.matmul(np.linalg.inv(np.matmul(A.T, A)) + lambda_ * np.eye(A.shape[1]),
                     np.matmul(A.T, b))  # compute the pseudo-inverse


def solve_null_space(A):
    assert isinstance(A, np.ndarray)
    assert len(A.shape) == 2, A.shape
    assert np.linalg.matrix_rank(A) == min(A.shape) - 1, 'null space is more than one dimension'
    return


def main():
    # make data to fit
    # # make up a curve
    domain = (-100, 100)
    # paras = [3.4, -1.5, -2.2, 3.1, 2.1, 1.1]    # parameters
    paras = [0.0, 1.0]  # parameters
    t = np.linspace(*domain, 1000)  # time
    A = np.zeros((len(t), len(paras)))  # observation-power matrix
    for i in range(len(paras)):
        A[:, i] = np.power(t, i)
    paras = np.array(paras).reshape(A.shape[1], 1)
    proj = np.matmul(A, paras)  # observations
    # # add noise
    amp = 100.0
    noise = np.random.random(proj.shape) - 0.5
    proj = proj + noise * amp

    plt.scatter(t, proj)
    # proj = np.sort(proj, axis=0)
    # plt.scatter(t, proj)
    plt.show()

    # solve linear_least_square
    paras_exp = line_fitting(domain=domain, observations=proj, num_power=A.shape[-1], method='ransac')
    proj_ = np.matmul(A, paras_exp)  # observation according to line fitting
    # y_ = np.squeeze(y_, axis=-1).reshape(t.shape[0], -1)
    print(paras, '\n', paras_exp)
    # plt.scatter(t, proj)
    # plt.show()
    plt.plot(t, proj_)
    plt.scatter(t, proj)
    plt.show()


if __name__ == "__main__":
    main()
