# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import json

import matplotlib.pyplot as plt
import numpy
import numpy as np


def netwon():
    print()
    x = 5
    data = numpy.zeros((1, 4))
    for i in range(1000):
        y, y1, y2 = func(x)
        data = np.concatenate([data, np.asarray([[x, y, y1, y2]])], axis=0)
        print(data[-1])
        if abs(y1) < 0.1:
            print(data[-1])
            break
        delta = 0.5 * -y2 / y1
        x += delta
    data = data[1:, :]
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def gassian_netwon():
    print()
    x = 5
    y_hat = 1000

    data = numpy.zeros((1, 4))
    for i in range(1000):
        y, y1, y2 = func(x)
        data = np.concatenate([data, np.asarray([[x, y, y1, y2]])], axis=0)
        print(data[-1])
        if abs(y_hat - y) < 0.001:
            print(data[-1])
            break
        delta = (y_hat - y) / y1 * 0.1
        x += delta
    data = data[2:, :]
    plt.scatter(data[:, 0], data[:, 1])
    plt.show()


def gradient_decent():
    print()
    x = 5
    y_hat = 1000

    data = numpy.zeros((1, 4))
    for i in range(1000):
        y, y1, y2 = func(x)
        data = np.concatenate([data, np.asarray([[x, y, y1, y2]])], axis=0)
        print(data[-1])
        if abs(y_hat - y) < 0.001:
            print(data[-1])
            break
        delta = y1 * 0.1 if y_hat > y else - y1 * 0.01
        x += delta
    data = data[2:, :]
    plt.scatter(data[:, 0], data[:, 1])
    low, up = np.min(data[:, 0]), np.max(data[:, 0])
    plt.xticks(np.arange(low, up, (up - low) / 10))
    plt.show()


def func(x):
    return x * x, 2 * x, 2


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.



def main():
    # netwon()
    # gassian_netwon()
    # gradient_decent()
    cout = {"a": 0,
            "b": 1}
    j = json.dumps(cout)
    print(j)
    with open('./01.json', 'w') as f:
        json.dump(cout, f)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
