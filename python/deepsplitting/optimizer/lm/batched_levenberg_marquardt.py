# - vangrad
# - armijo
# - damping

import torch
import logging

import deepsplitting.utils.global_config as global_config
from deepsplitting.optimizer.base import BaseOptimizer


# L(W) = (0.5) * norm(f(W;X) - Y)^2
# L_lin(d) = (0.5) * norm(J*d + f(W;X) - Y)^2 = (0.5) * norm(J*d - (Y - f(W;X)))^2

class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

        if not isinstance(self.net.criterion, torch.nn.MSELoss):
            raise ValueError("Only works with least squares loss.")

        self.N = N
        self.iteration = 1

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

        self.iteration = 1

    def step(self, inputs, labels):
        batch_size = global_config.cfg.training_batch_size
        indices, subsample_size, max_steps = self.fullbatch_subindex_init(batch_size,
                                                                          self.hyperparams.subsample_factor,
                                                                          self.N)

        loss = list()

        loss.append(self.loss_chunked(inputs, labels))

        for i, index in enumerate(self.batches(indices, batch_size), 1):
            subindex = self.rand_subindex(batch_size, subsample_size)

            self.step_batched(inputs, labels, index, subindex)

            current_loss = self.loss_chunked(inputs, labels)
            loss.append(current_loss)

            logging.info("{} (Batch step [{}/{}]): Data loss = {:.6f}".format(
                type(self).__module__, i, max_steps, current_loss))

            self.iteration += 1

        return loss

    def loss_chunked(self, inputs, labels):
        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor)
        return self.net.criterion(y, labels)

    def step_batched(self, inputs, labels, index, subindex):
        raise NotImplementedError()


class Optimizer_damping(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_damping, self).__init__(net, N, hyperparams)

        self.M = self.hyperparams.M

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer_damping, self).init(inputs, labels, initializer, parameters)

        self.M = self.hyperparams.M

    def cg_step(self, index, subindex, inputs, labels):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        B = J1.t().matmul(torch.reshape(labels[index] - y_batch, (-1, 1)))

        x = torch.zeros(J2.size(1), 1, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + self.hyperparams.M * p

        return self.cg_solve(x, A, B)

    def step_batched(self, inputs, labels, index, subindex):
        loss = self.loss_chunked(inputs, labels)

        while True:
            new_params = self.cg_step(index, subindex, inputs, labels)

            saved_params = self.load(new_params)
            loss_new = self.loss_chunked(inputs, labels)

            if loss < loss_new:
                self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                self.restore(saved_params)
            else:
                self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                break


class Optimizer_armijo(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_armijo, self).__init__(net, N, hyperparams)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer_armijo, self).init(inputs, labels, initializer, parameters)

    def cg_step(self, inputs, labels, index, subindex, delta):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        R = torch.reshape(labels[index] - y_batch, (-1, 1))
        B = J1.t().matmul(R)

        x = torch.zeros(J2.size(1), 1, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        mu = torch.norm(R) ** delta

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + mu * p

        new_params, step = self.cg_solve(x, A, B, return_step=True)

        # Required for armijo.
        dderiv = B.t().matmul(step)

        return new_params, step, B, dderiv

    def step_batched(self, inputs, labels, index, subindex):
        delta = self.hyperparams.delta
        eta = self.hyperparams.eta

        beta = self.hyperparams.beta
        gamma = self.hyperparams.gamma

        loss_current = self.loss_chunked(inputs, labels)

        new_params, step, B, dderiv = self.cg_step(inputs, labels, index, subindex, delta)

        current_params = self.load(new_params)
        loss_new = self.loss_chunked(inputs, labels)

        if not (torch.norm(loss_new) <= eta * torch.norm(loss_current)):
            self.load(current_params)
            self.armijo(inputs, labels, step, beta, gamma, loss_current, dderiv)

    def armijo(self, inputs, labels, step, beta, gamma, loss_current, dderiv):
        k = 1

        while True:
            sigma = (beta ** k) / beta

            if self.check_armijo(inputs, labels, step, sigma, gamma, loss_current, dderiv):
                break

            k = k + 1

    def check_armijo(self, inputs, labels, step, sigma, gamma, loss_current, dderiv):
        current_params = self.save_params()

        new_params = self.vec_to_params_update(sigma * step, from_numpy=False)
        self.restore_params(new_params)

        loss_new = self.loss_chunked(inputs, labels)

        if loss_new - loss_current <= sigma * gamma * dderiv:
            return True

        self.restore_params(current_params)

        return False


class Optimizer_vanstep(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_vanstep, self).__init__(net, N, hyperparams)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer_vanstep, self).init(inputs, labels, initializer, parameters)

    def cg_step(self, inputs, labels, index, subindex, delta):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        R = torch.reshape(labels[index] - y_batch, (-1, 1))
        B = J1.t().matmul(R)

        x = torch.zeros(J2.size(1), 1, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        mu = torch.norm(R) ** delta

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + mu * p

        new_params, step = self.cg_solve(x, A, B, return_step=True)

        # Required for armijo.
        dderiv = B.t().matmul(step)

        return new_params, step, B, dderiv

    def step_batched(self, inputs, labels, index, subindex):
        delta = self.hyperparams.delta
        eta = self.hyperparams.eta

        loss_current = self.loss_chunked(inputs, labels)

        new_params, step, B, dderiv = self.cg_step(inputs, labels, index, subindex, delta)

        self.load(new_params)
        loss_new = self.loss_chunked(inputs, labels)

        if not (torch.norm(loss_new) <= eta * torch.norm(loss_current)):
            sigma = 1 / self.iteration

            new_params = self.vec_to_params_update(sigma * step, from_numpy=False)
            self.load(new_params)
