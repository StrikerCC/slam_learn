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
import os
import cv2
import numpy as np
# from functions import quadratic, tuplti


def least_square(A, b):
    x = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), b)  # compute the pseudo-inverse
    return x


def ransac(A, b):
    iter_max = 1000
    num_ele_min = A.shape[-1]
    x_best = np.zeros((A.shape[-1], 1))
    error_best = rms_error(np.dot(A, x_best), b)
    for it in range(iter_max):
        index_smallest = np.random.choice(len(A), num_ele_min, replace=False)
        A_smallest, b_smallest = A[index_smallest], b[index_smallest]
        if np.linalg.matrix_rank(A_smallest) < num_ele_min:
            continue
        x_cur = np.matmul(np.linalg.inv(A_smallest), b_smallest)
        error_cur = rms_error(np.dot(A, x_cur), b)
        if error_cur < error_best:
            x_best, error_best = x_cur, error_cur
        if error_best < 0.00001:
            print(it, 'iter jump out')
            return x_best
    return x_best


def rms_error(data_a, data_b):
    assert isinstance(data_a, np.ndarray) and isinstance(data_b, np.ndarray)
    assert data_a.shape == data_b.shape
    assert len(data_a.shape) == len(data_b.shape) == 2

    # transpose the data if number of rows is bigger than number of cols
    if data_a.shape[0] < data_a.shape[1]:
        data_a, data_b = data_a.T, data_b.T
    # diff = data_a - data_b
    # diff = np.linalg.norm(diff, axis=1)
    # diff = np.mean(diff, axis=0)
    return np.mean(np.linalg.norm(data_a - data_b, axis=1), axis=0)


# def non_linear_least_square():


def qrdecode(img):
    qrcode = cv2.QRCodeDetector()
    result, points = qrcode.detect(img)
    print(result)
    print(points)
    # print(code)


def main():
    file_paths = os.listdir('./data/qrcode')
    for file_path in file_paths:
        file_path = './data/qrcode/' + file_path

        img = cv2.imread(file_path)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # corners = cv2.goodFeaturesToTrack(img, maxCorners=4, qualityLevel=0.1, minDistance=100)
        # corners = np.squeeze(corners, axis=1)
        #
        # print(corners.shape)
        # for p in corners:
        #     p = tuple(p.astype(int).tolist())
        #     img = cv2.circle(img, p, radius=10, color=(0, 0, 1))

        # contours, _ = cv2.findContours(gray, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)
        # print(len(contours), contours[0].shape)

        # img_ = cv2.drawContours(img, contours, contourIdx=-1, color=(0, 0, 255), thickness=10)

        cv2.namedWindow(file_path, 0)
        cv2.imshow(file_path, gray)
        cv2.waitKey(0)
        cv2.destroyWindow(file_path)

        print(file_path)
        qrdecode(gray)


if __name__ == '__main__':
    main()

