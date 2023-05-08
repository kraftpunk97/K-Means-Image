import random
import numpy as np
import numpy.matlib


def init_centroids(X, K):
    c = random.sample(list(X), K)
    return c


def closest_centroids(X, c):
    K = np.size(c, 0)
    arr = np.empty((np.size(X, 0), 1))
    for i in range(0, K):
        y = c[i]
        temp = np.ones((np.size(X, 0), 1))*y
        b = np.power(np.subtract(X, temp), 2)
        a = np.sum(b, axis=1)
        a = np.asarray(a)
        a.resize((np.size(X, 0), 1))
        arr = np.append(arr, a, axis=1)
    arr = np.delete(arr, 0, axis=1)
    idx = np.argmin(arr, axis=1)
    return idx


def compute_centroids(X, idx, K):
    n = np.size(X, 1)
    centroids_ = np.zeros((K, n))
    for i in range(0, K):
        ci = idx == i
        ci = ci.astype(int)
        total_number = sum(ci)
        ci.resize((np.size(X, 0), 1))
        total_matrix = np.matlib.repmat(ci, 1, n)
        total = np.multiply(X, total_matrix)
        centroids_[i] = (1 / total_number) * np.sum(total, axis=0)
    return centroids_


def run_kMean(X, initial_centroids, max_iters):
    m = np.size(X, 0)
    n = np.size(X, 1)
    K = np.size(initial_centroids, 0)
    centroids = initial_centroids
    idx = np.zeros((m, 1))
    for i in range(1, max_iters):
        if i % 5 == 0 or i == 1:
            print("Running iteration: {}/{}".format(i, max_iters))
        idx = closest_centroids(X, centroids)
        centroids = compute_centroids(X, idx, K)
    return centroids, idx
