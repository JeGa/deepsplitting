import torch
import numpy as np
from scipy.optimize import minimize

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
        N, c = y.size()
        lin = y + torch.from_numpy(np.reshape(J.dot(d), (N, c)))

        loss = self.net.criterion(lin, y_train)

        reg = 0.5 * mu * np.sum(d ** 2)

        return loss + reg

    def init(self, debug=False):
        super(Optimizer, self).init_parameters(debug)

        self.mu = self.hyperparams.mu_min


def minimize_linearized_penalty_nll(J, y, y_train, mu, loss):
    d0 = np.ones(J.shape[1])

    def linearized_penalty(d):
        d = np.expand_dims(d, 1)

        f = y + np.reshape(J.dot(d), y.shape)
        L = loss(torch.from_numpy(f), torch.from_numpy(y_train)).item()

        reg = + 0.5 * mu * np.sum(np.power(d, 2))

        return L + reg

    def gradient_linearized_penalty(d):
        d = np.expand_dims(d, 1)

        N, c = y.shape

        grad = np.zeros(d.shape)

        for i in range(N):
            from_sample = i * c
            to_sample = from_sample + c

            Ji = J[from_sample:to_sample, :]

            yi = np.expand_dims(y[i, :], 1)
            yi_train = y_train[i]
            fi = yi + Ji.dot(d)

            fit = torch.from_numpy(fi.T)
            fit.requires_grad = True

            L = loss(fit, torch.from_numpy(np.expand_dims(yi_train, 0)))
            L.backward()
            gL = fit.grad.data.numpy().T

            grad = grad + Ji.T.dot(gL)

        return (grad + mu * d).squeeze()

    result = minimize(linearized_penalty, d0, jac=gradient_linearized_penalty,
                      options={'gtol': 1e-05, 'norm': np.inf, 'eps': 1.4901161193847656e-08, 'maxiter': None,
                               'disp': False, 'return_all': False})

    d = result.x

    return d


# TODO: If not 0.5 * loss.
def minimize_linearized_penalty_ls(J, y, y_train, mu, loss):
    if loss.size_average is True:
        N = y_train.shape[0]
    else:
        N = 1

    return np.linalg.inv(mu * N * np.eye(J.shape[1]) + J.T.dot(J)).dot(J.T.dot(np.reshape(y_train - y, (-1, 1))))
