import numpy as np
import torch

from deepsplitting.optimizer.base import BaseOptimizer
from deepsplitting.optimizer.base import Hyperparams


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

        if not isinstance(self.net.criterion, torch.nn.MSELoss):
            raise ValueError("Only works with least squares loss.")

        self.M = self.hyperparams.M

    def step(self, inputs, labels):
        y = self.net(inputs)
        J = self.jacobian(y)

        L = self.net.criterion(y, labels)

        while True:
            s = self.levmarq_step(J, y.detach().numpy(), labels.numpy(), self.M)
            s = np.float32(s) # TODO: float
            new_params = self.vec_to_params_update(s)

            old_params = self.save_params()
            self.restore_params(new_params)

            L_new = self.net.loss(inputs, labels)

            if L < L_new:
                self.M = self.M * self.hyperparams.factor
                self.restore_params(old_params)
            else:
                self.M = self.M / self.hyperparams.factor
                break

        return L.item(), L_new.item()

    def levmarq_step(self, J, y, y_train, M):
        r = np.reshape(y_train - y, (-1, 1))

        A = (J.T.dot(J) + M * np.eye(J.shape[1]))
        B = J.T.dot(r)

        return np.linalg.solve(A, B)

    def start_eval(self, inputs, labels):
        return self.net.loss(inputs, labels)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

        self.M = self.hyperparams.M
