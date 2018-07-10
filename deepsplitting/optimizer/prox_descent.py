import torch
import numpy as np
import logging

from .base import BaseOptimizer
from .base import Hyperparams


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams=Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3)):
        super().__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.minimize_linearized_penalty = minimize_linearized_penalty_nll
        elif isinstance(self.net.criterion, torch.nn.MSELoss):
            self.minimize_linearized_penalty = minimize_linearized_penalty_ls
        else:
            raise AttributeError('Loss not supported: No minimize linearized penalty function.')

        self.mu = self.hyperparams.mu_min

    def step(self, inputs, labels):
        y = self.net(inputs)
        J = self.jacobian(y)

        while True:
            d = self.minimize_linearized_penalty(J, y.detach().numpy(), labels.numpy(), self.mu, self.net.criterion)
            new_params = self.vec_to_params_update(d)

            # With current parameters.
            L_current = self.net.loss(inputs, labels)

            # Update parameters.
            old_params = self.save_params()
            self.restore_params(new_params)

            # With new parameters.
            L_new = self.net.loss(inputs, labels)

            L_new_linearized = self.loss_linearized(labels, J, y, d, self.mu)

            diff_real = L_current - L_new
            diff_linearized = L_current - L_new_linearized

            if diff_real >= self.hyperparams.sigma * diff_linearized:
                self.mu = max(self.hyperparams.mu_min, self.mu / self.hyperparams.tau)
                break
            else:
                self.mu = self.hyperparams.tau * self.mu
                self.restore_params(old_params)

        return L_current.item(), L_new.item()

    def loss_linearized(self, y_train, J, y, d, mu):
        N, c = y_train.size()
        lin = y + torch.from_numpy(np.reshape(J.dot(d), (N, c)))

        loss = self.net.criterion(lin, y_train)

        reg = 0.5 * mu * np.sum(d ** 2)

        return loss + reg


def minimize_linearized_penalty_nll(J, y, y_train, mu, loss):
    raise NotImplementedError


# TODO: What if not 0.5 * loss.
def minimize_linearized_penalty_ls(J, y, y_train, mu, loss):
    if loss.size_average is True:
        N = y_train.shape[0]
    else:
        N = 1

    return np.linalg.inv(mu * N * np.eye(J.shape[1]) + J.T.dot(J)).dot(J.T.dot(np.reshape(y_train - y, (-1, 1))))
