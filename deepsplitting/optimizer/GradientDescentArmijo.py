import torch
from collections import namedtuple
from .Base import BaseOptimizer

hyperparams = namedtuple('hyperparams', ['beta', 'gamma'])


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams=hyperparams(beta=0.5, gamma=10 ** -4)):
        super(Optimizer, self).__init__(net, hyperparams)

    def step(self, eval):
        params = [p for p in self.net.parameters() if p.grad is not None]
        grads = [p.grad.data for p in params]

        step_direction = [-1 * p.grad.data for p in params]

        self._armijo_step(params, grads, step_direction, eval)

    def _armijo_step(self, params, grads, step_direction, eval):
        k = 1

        while True:
            sigma = (self.hyperparams.beta ** k) / self.hyperparams.beta

            if self._check_armijo(sigma, params, grads, step_direction, eval):
                break

            k = k + 1

    def _check_armijo(self, sigma, params, grads, step_direction, eval):
        current_loss = eval()
        current_params = [p.clone() for p in self.net.parameters()]

        for p, s in zip(params, step_direction):
            p.data.add_(sigma, s)

        new_loss = eval()

        slope = sum([grad.mul(s).sum() for grad, s in zip(grads, step_direction)])

        if new_loss - current_loss <= sigma * self.hyperparams.gamma * slope:
            return True

        with torch.no_grad():
            for old_p, new_p in zip(current_params, self.net.parameters()):
                new_p.copy_(old_p)
        return False
