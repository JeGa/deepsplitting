# TODO: No Regularizer currently!

import numpy as np
import torch

from .base import BaseOptimizer
from .base import Hyperparams


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams=Hyperparams(beta=0.5, gamma=10 ** -4, M=0.001, factor=10, rho=1)):
        super(Optimizer, self).__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.primal1_loss = primal1_nll
        elif type(self.net.criterion) is torch.nn.MSELoss:
            self.primal1_loss = primal1_ls
        else:
            raise AttributeError('Loss not supported: No primal1 update function.')

        self.lam = torch.ones(N, self.net.output_dim)
        self.v = 0.1 * torch.randn(N, self.net.output_dim)

    def step(self, inputs, labels):
        self.primal2_levmarq(inputs, labels)

        self.primal1(inputs, labels)

        self.dual(inputs)

        self.hyperparams.rho = min(1, self.hyperparams.rho + 0.1)

        L, L_data, _ = self.augmented_lagrangian(inputs, labels)

        print("Lagrangian = {}, Data Loss = {}".format(L, L_data))

        return L_data

    def augmented_lagrangian(self, inputs, labels, params=None):
        if params is not None:
            self.restore_params(params)

        y = self.net(inputs)
        data_loss = self.net.criterion(y, labels)

        loss = self.net.criterion(self.v, labels)
        constraint = y - self.v
        constraint_norm = torch.pow(constraint, 2).sum()

        L = loss + torch.mul(self.lam, constraint).sum() + (self.hyperparams.rho / 2) * constraint_norm

        return L, data_loss, y

    def primal2_levmarq(self, inputs, labels):
        i = 0
        max_iter = 1

        while True:
            i += 1

            L, _, y = self.augmented_lagrangian(inputs, labels)
            J = self.jacobian(y)

            while True:
                self.levmarq_step(J, y)

                L_new, _, _ = self.augmented_lagrangian(inputs, labels)

                if L < L_new:
                    self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                else:
                    self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                    break

            if i == max_iter:
                break

    def levmarq_step(self, J, y):
        r = self.v - self.lam / self.hyperparams.rho - y
        r = torch.reshape(r, (-1,)).detach().numpy()

        s = np.linalg.solve((J.T.dot(J) + self.hyperparams.M * np.eye(J.shape[1], dtype=np.float32)), J.T.dot(r))

        start_index = 0
        for p in self.net.parameters():
            with torch.no_grad():
                p.add_(torch.reshape(torch.from_numpy(s[start_index:start_index + p.numel()]), p.size()))
            start_index += p.numel()

    def primal1(self, inputs, labels):
        y = self.net(inputs)

        self.v = self.primal1_loss(y, self.lam, labels, self.hyperparams.rho)

    def dual(self, inputs):
        y = self.net(inputs)

        self.lam = self.lam + self.hyperparams.rho * (y - self.v)


def primal1_ls(y, lam, y_train, rho):
    C = 1 / y_train.size(0)
    return (C * y_train + rho * y + lam) / (C + rho)


def primal1_nll():
    return 0
