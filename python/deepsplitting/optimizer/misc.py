import numpy as np
from torch.nn import functional as F


def prox_cross_entropy(q, tau, y):
    q = q.squeeze()
    rho = 1 / tau

    c = q.shape[0]

    I = np.eye(c)
    b = q + I[:, y] / rho

    def f(t):
        p = b - t
        return np.sum(lambertw_exp(p), 0) - (1 / rho)

    def V(t):
        return lambertw_exp(t)

    def dV(t):
        return V(t) / (1 + V(t))

    def df(t):
        return -np.sum(dV(b - t), 0)

    t = 0.5
    t = newton_nls(t, f, df)
    lam = V(b - t) * rho
    x = q - (lam - I[:, y]) / rho

    return x


def lambertw_exp(X):
    """
    :param X: Numpy array of shape (c,1).
    :return: Numpy array of shape (c,1).
    """
    # From https://github.com/foges/pogs/blob/master/src/include/prox_lib.h
    C1 = X > 700
    C2 = np.logical_and(np.logical_not(C1), X < 0)
    C3 = np.logical_and(np.logical_not(C1), X > 1.098612288668110)

    W = np.copy(X)
    log_x = np.log(X[C1])
    W[C1] = -0.36962844 + X[C1] - 0.97284858 * log_x + 1.3437973 / log_x
    p = np.sqrt(2.0 * (np.exp(X[C2] + 1.) + 1))
    W[C2] = -1.0 + p * 1.0 + p * (-1.0 / 3.0 + p * (11.0 / 72.0))

    W[C3] = W[C3] - np.log(W[C3])

    for i in range(10):
        e = np.exp(W[np.logical_not(C1)])
        t = W[np.logical_not(C1)] * e - np.exp(X[np.logical_not(C1)])
        p = W[np.logical_not(C1)] + 1.
        t = t / (e * p - 0.5 * (p + 1.0) * t / p)
        W[np.logical_not(C1)] = W[np.logical_not(C1)] - t

        if np.max(np.abs(t)) < np.min(np.spacing(np.float32(1)) * (1 + np.abs(W[np.logical_not(C1)]))):
            break

    return W


def newton_nls(init, f, df):
    x = init

    for i in range(30):
        s = -f(x) / df(x)
        sigma = 1

        x = x + sigma * s

        if np.abs(f(x)) < np.spacing(np.float32(1)):
            break

    return x


def bias_to_end(J, ls):
    """
    For debugging.

    ls = net.layer_size.
    """
    Jw = []
    Jb = []

    from_index = 0

    for i in range(len(ls) - 1):
        to_index = from_index + ls[i] * ls[i + 1]

        Jw.append(J[:, from_index:to_index])
        from_index = to_index
        to_index = from_index + ls[i + 1]

        Jb.append(J[:, from_index:to_index])
        from_index = to_index

    return np.concatenate(Jw + Jb, axis=1)


def grad_relu(x):
    """
    :param x: (N, d)
    :returns: J in matrix form (N, d).
    """
    return (x > 0.0).double()


def grad_sigmoid(x):
    """
    :param x: (N, d)
    :returns: J in matrix form (N, d).
    """
    return F.sigmoid(x).mul(1 - F.sigmoid(x))
