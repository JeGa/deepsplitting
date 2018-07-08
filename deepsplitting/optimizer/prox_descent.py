import torch
import numpy as np

from .base import BaseOptimizer
from .base import Hyperparams


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams=Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3)):
        super().__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.minimize_linearized_penalty = minimize_linearized_penalty_nll
            self.loss_linearized = loss_linearized_nll
        elif isinstance(self.net.criterion, torch.nn.MSELoss):
            self.minimize_linearized_penalty = minimize_linearized_penalty_ls
            self.loss_linearized = loss_linearized_ls
        else:
            raise AttributeError('Loss not supported: No minimize linearized function.')

        self.mu = self.hyperparams.mu_min

    def step(self, inputs, labels):
        y = self.net(inputs)
        J = self.jacobian(y)

        while True:
            # Step.
            d = self.minimize_linearized_penalty(J, y, labels, self.mu, self.net.criterion)
            new_params = self.vec_to_params_update(d)

            L_current = self.net.loss(inputs, labels)

            # Update weights.
            old_params = self.save_params()
            self.restore_params(new_params)

            L_new = self.net.loss(inputs, labels)

            L_new_linearized = self.loss_linearized(labels, J, y, d, self.hyperparams.mu)

    def loss_linearized(self, y_train, J, y, d, mu):
        pass


def minimize_linearized_penalty_nll(J, y, y_train, mu, loss):
    raise NotImplementedError


def loss_linearized_nll(labels, J, y, d, mu):
    raise NotImplementedError


# TODO: What if not 0.5 * loss.
def minimize_linearized_penalty_ls(J, y, y_train, mu, loss):
    if loss.size_average is True:
        N = 1 / y_train.size(0)
    else:
        N = 1

    d = np.linalg.inv(mu * N * np.eye(J.shape[1]) + J.T.dot(J)).dot(J.T.dot(np.reshape(y_train - y, (-1, 1))))
    return d


def loss_linearized_ls(labels, J, y, d, mu):
    pass
