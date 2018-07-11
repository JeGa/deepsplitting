# TODO: No regularizer currently!

import numpy as np
import torch
import logging

from .base import BaseOptimizer
from .base import Hyperparams
from .misc import prox_cross_entropy


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams=Hyperparams(M=0.001, factor=10, rho=1)):
        super(Optimizer, self).__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.primal1_loss = primal1_nll
        elif isinstance(self.net.criterion, torch.nn.MSELoss):
            self.primal1_loss = primal1_ls
        else:
            raise AttributeError('Loss not supported: No primal1 update function.')

        self.N = N

        self.init_variables()

    def init_variables(self, debug=False):
        self.lam = torch.ones(self.N, self.net.output_dim, dtype=torch.double)

        if debug:
            self.v = torch.zeros(self.N, self.net.output_dim, dtype=torch.double)
        else:
            self.v = 0.1 * torch.randn(self.N, self.net.output_dim, dtype=torch.double)

    def init(self, debug=False):
        super(Optimizer, self).init_parameters(debug)

        self.init_variables(debug)

    def step(self, inputs, labels):
        L_data_current = self.eval(inputs, labels)

        self.primal2_levmarq(inputs, labels)

        self.primal1(inputs, labels)

        self.dual(inputs)

        self.hyperparams.rho = min(1, self.hyperparams.rho + 0.1)

        L_data_new = self.eval(inputs, labels)

        return L_data_current.item(), L_data_new.item()

    def eval(self, inputs, labels, print=False):
        L, L_data, loss, constraint_norm, _ = self.augmented_lagrangian(inputs, labels)

        if print:
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

                logging.debug("levmarq_step: L={:.2f}, L_new={:.2f}, M={}".format(L, L_new, self.hyperparams.M))

                if L < L_new:
                    self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                else:
                    self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                    self.restore_params(params)
                    break

            if i == max_iter:
                break

    def levmarq_step(self, J, y):
        r = self.v - self.lam / self.hyperparams.rho - y
        r = torch.reshape(r, (-1, 1)).data.numpy()

        A = J.T.dot(J) + self.hyperparams.M * np.eye(J.shape[1])
        B = J.T.dot(r)

        s = np.linalg.solve(A, B)

        param_list = self.vec_to_params_update(s)

        return param_list

    def primal1(self, inputs, labels):
        y = self.net(inputs)

        self.v = self.primal1_loss(y, self.lam, labels, self.hyperparams.rho, self.net.criterion)

    def dual(self, inputs):
        y = self.net(inputs)

        self.lam = self.lam + self.hyperparams.rho * (y - self.v)


def primal1_ls(y, lam, y_train, rho, loss):
    C = 2 * loss.C
    if loss.size_average is True:
        C = C * 1 / y_train.size(0)

    return (C * y_train + rho * y + lam) / (C + rho)


def primal1_nll(y, lam, y_train, rho, loss):
    z = torch.zeros(y.size(), dtype=torch.double)

    r = y + lam / rho

    for i in range(y_train.size(0)):
        cls = y_train[i]
        z[i] = torch.from_numpy(prox_cross_entropy(np.expand_dims(r[i].detach().numpy(), 1), 1 / rho, cls.item()))

    return z
