# - vangrad
# - armijo
# - damping

import torch

from deepsplitting.optimizer.base import BaseOptimizer


# (0.5) * norm(f(W;X) - Y)^2

class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

        if not isinstance(self.net.criterion, torch.nn.MSELoss):
            raise ValueError("Only works with least squares loss.")

        self.M = self.hyperparams.M

    def step(self, inputs, labels):
        pass

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

        self.M = self.hyperparams.M
