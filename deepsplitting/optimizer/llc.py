# TODO: No Regularizer currently!

import numpy as np
import torch
import logging

from .base import BaseOptimizer
from .base import Hyperparams

M_INIT = 0.001


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams=Hyperparams(M=M_INIT, factor=10, rho=1)):
        super(Optimizer, self).__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.primal1_loss = primal1_nll
        elif isinstance(self.net.criterion, torch.nn.MSELoss):
            self.primal1_loss = primal1_ls
        else:
            raise AttributeError('Loss not supported: No primal1 update function.')

        self.lam = torch.ones(N, self.net.output_dim, dtype=torch.double)
        # self.v = 0.1 * torch.randn(N, self.net.output_dim)
        self.v = torch.zeros(N, self.net.output_dim, dtype=torch.double)

        def init(submodule):
            if type(submodule) == torch.nn.Linear:
                # torch.nn.init.normal_(submodule.weight)
                # torch.nn.init.normal_(submodule.bias)

                # submodule.weight.data.mul_(0.1)
                # submodule.bias.data.mul_(0.1)

                submodule.weight.data.fill_(1)
                submodule.bias.data.fill_(1)

        net.apply(init)

    def step(self, inputs, labels):
        self.primal2_levmarq(inputs, labels)  # TODO: This is correct.

        self.primal1(inputs, labels)  # TODO: This is correct.

        self.dual(inputs)

        self.hyperparams.rho = min(1, self.hyperparams.rho + 0.1)

        L_data = self.eval_print(inputs, labels)

        return L_data

    def eval_print(self, inputs, labels):
        L, L_data, loss, constraint_norm, _ = self.augmented_lagrangian(inputs, labels)

        logging.info(
            "Data Loss = {:.8f}, Loss = {:.8f}, Constraint norm = {:.8f}, Lagrangian = {:.8f}".format(
                L_data, loss, constraint_norm, L))

        return L_data

    def augmented_lagrangian(self, inputs, labels, params=None):
        if params is not None:
            saved_params = self.save_params()
            self.restore_params(params)

        y = self.net(inputs)
        data_loss = self.net.criterion(y, labels)

        loss = self.net.criterion(self.v, labels)
        constraint = y - self.v
        constraint_norm = torch.pow(constraint, 2).sum()

        L = loss + torch.mul(self.lam, constraint).sum() + (self.hyperparams.rho / 2) * constraint_norm

        if params is not None:
            self.restore_params(saved_params)

        return L, data_loss, loss, constraint_norm, y

    def primal2_levmarq(self, inputs, labels):
        i = 0
        max_iter = 1

        while True:
            i += 1

            L, _, _, _, y = self.augmented_lagrangian(inputs, labels)
            J = self.jacobian(y)

            while True:
                params = self.levmarq_step(J, y)

                L_new, _, _, _, _ = self.augmented_lagrangian(inputs, labels, params)

                logging.info("levmarq_step: L={:.2f}, L_new={:.2f}, M={}".format(L, L_new, self.hyperparams.M))

                if L < L_new:
                    self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                else:
                    # if self.hyperparams.M > M_INIT:
                    self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                    self.restore_params(params)
                    break

            if i == max_iter:
                break

    def levmarq_step(self, J, y):
        r = self.v - self.lam / self.hyperparams.rho - y
        r = torch.reshape(r, (-1, 1)).data.numpy()

        Jw = []
        Jb = []

        from_index = 0
        ls = self.net.layer_size

        for i in range(len(ls) - 1):
            to_index = from_index + ls[i] * ls[i + 1]

            Jw.append(J[:, from_index:to_index])
            from_index = to_index
            to_index = from_index + ls[i + 1]

            Jb.append(J[:, from_index:to_index])
            from_index = to_index

        Jr = np.concatenate(Jw + Jb, axis=1)

        # J = Jr

        A = J.T.dot(J) + self.hyperparams.M * np.eye(J.shape[1])
        B = J.T.dot(r)

        # print(r)
        # print(A)
        # print(B)

        s = np.linalg.solve(A, B)

        param_list = []

        start_index = 0
        for p in self.net.parameters():
            with torch.no_grad():
                params = torch.from_numpy(s[start_index:start_index + p.numel()])
                params_rs = torch.reshape(params, p.size())
                param_list.append(p + params_rs)

            start_index += p.numel()

        return param_list

    def primal1(self, inputs, labels):
        y = self.net(inputs)

        self.v = self.primal1_loss(y, self.lam, labels, self.hyperparams.rho)

    def dual(self, inputs):
        y = self.net(inputs)

        self.lam = self.lam + self.hyperparams.rho * (y - self.v)


def primal1_ls(y, lam, y_train, rho):
    C = 1  # / y_train.size(0)
    return (C * y_train + rho * y + lam) / (C + rho)


def primal1_nll(y, lam, y_train, rho):
    z = torch.zeros(y.size())

    r = y + lam / rho

    for i in range(y_train.size(0)):
        cls = y_train[i]
        z[i] = torch.from_numpy(prox_cross_entropy(np.expand_dims(r[i].detach().numpy(), 1), 1 / rho, cls.item()))

    return z


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


# TODO: Checked.
# lambertw_exp(np.array([[-0.576565155510187], [0.586943687333602]]))
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

        if np.max(np.abs(t)) < np.min(np.spacing(np.float64(1)) * (1 + np.abs(W[np.logical_not(C1)]))):
            break

    return W


def newton_nls(init, f, df):
    x = init

    for i in range(30):
        s = -f(x) / df(x)
        sigma = 1

        x = x + sigma * s

        if np.abs(f(x)) < np.spacing(np.float64(1)):
            break

    return x
