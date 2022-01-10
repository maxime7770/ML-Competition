import numpy as np


def dist(x, y):
    return np.sqrt((x[0]-y[0])**2+(x[1]-y[1])**2)


def perimeter(li):
    res = 0
    for i in range(len(li)-1):
        res += dist(li[i], li[i+1])
    res += dist(li[len(li)], li[0])
    return res
