from deepsplitting.optimizer.base import BaseOptimizer
import deepsplitting.utils.global_progressbar as gp


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

    def step(self, inputs, labels):
        loss = self.net.loss(inputs, labels)
        loss.backward()

        params = [p for p in self.net.parameters() if p.grad is not None]
        grads = [p.grad.data for p in params]

        step_direction = [-1 * p.grad.data for p in params]

        self._armijo_step(params, grads, step_direction, inputs, labels)

        return [loss.item()], [self.net.loss(inputs, labels).item()]

    def _armijo_step(self, params, grads, step_direction, inputs, labels):
        current_loss = self.net.loss(inputs, labels)

        k = 1

        while True:
            sigma = (self.hyperparams.beta ** k) / self.hyperparams.beta

            if self._check_armijo(sigma, params, grads, step_direction, inputs, labels, current_loss):
                break

            k = k + 1

        gp.bar.next_batch(dict(dataloss=current_loss))

    def _check_armijo(self, sigma, params, grads, step_direction, inputs, labels, current_loss):
        current_params = self.save_params()

        for p, s in zip(params, step_direction):
            p.data.add_(sigma, s)

        new_loss = self.net.loss(inputs, labels)

        slope = sum([grad.mul(s).sum() for grad, s in zip(grads, step_direction)])

        if new_loss - current_loss <= sigma * self.hyperparams.gamma * slope:
            return True

        self.restore_params(current_params)

        return False

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)
