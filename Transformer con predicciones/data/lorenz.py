import numpy as np
import pandas as pd



def  lorenzData(time=1.6, stepsize=0.02, N=30):
    np.random.seed(1)
    n = 3 * N
    m = round(time / stepsize)
    X = np.zeros((n, m))
    X[:, 0] = np.random.rand(1, n)
    C = 0.1
    for i in range(m - 1):
        X[0, i + 1] = X[0, i] + stepsize * (10 * (X[1, i] - X[0, i]) + C * X[0 + (N - 1) * 3, i])
        X[1, i + 1] = X[1, i] + stepsize * (20 * X[0, i] - X[1, i] - X[0, i] * X[2, i])
        X[2, i + 1] = X[2, i] + stepsize * (-8/3 * X[2, i] + X[0, i] * X[1, i])
        for j in range(1, N):
            X[0 + 3 * j, i + 1] = X[0 + 3 * j, i] + stepsize * (10 * (X[1 + 3 * j, i] - X[0 + 3 * j, i]) + C * X[0 + 3 * (j - 1), i])
            X[1 + 3 * j, i + 1] = X[1 + 3 * j, i] + stepsize * (20 * X[0 + 3 * j, i] - X[1 + 3 * j, i] - X[0 + 3 * j, i] * X[2 + 3 * j, i])
            X[2 + 3 * j, i + 1] = X[2 + 3 * j, i] + stepsize * (-8 / 3 * X[2 + 3 * j, i] + X[0 + 3 * j, i] * X[1 + 3 * j, i])

    return X.T
