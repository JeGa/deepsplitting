import numpy as np
from torch.nn import functional as F
from scipy.special import lambertw


def prox_cross_entropy(q, tau, y):
    q = q.squeeze()
    rho = 1 / tau

    Iv = np.eye(q.shape[0])[:, y]

    b = q + Iv / rho

    def V(t):
        return lambertw_exp(t)

    def dV(t):
        return V(t) / (1 + V(t))

    def f(t):
        return np.sum(V(b - t)) - (1 / rho)

    def df(t):
        return -np.sum(dV(b - t))

    t = 0.5
    t = newton_nls(t, f, df)

    lam = V(b - t) * rho
    x = q - (lam - Iv) / rho

    return x


def newton_nls(init, f, df):
    x = init

    for i in range(30):
        s = -f(x) / df(x)
        sigma = 1

        x = x + sigma * s

        #if np.abs(f(x)) < np.spacing(np.float64(1)):
        #    break

    return x


def lambertw_exp_scipy(z):
    return lambertw(np.exp(z)).real


lambertw_exp = lambertw_exp_scipy


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
    return (x > 0.0).float()


def grad_sigmoid(x):
    """
    :param x: (N, d)
    :returns: J in matrix form (N, d).
    """
    return F.sigmoid(x).mul(1 - F.sigmoid(x))
