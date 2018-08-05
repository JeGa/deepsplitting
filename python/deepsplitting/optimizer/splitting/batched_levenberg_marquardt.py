import torch

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.splitting.batched_primal2 import Optimizer_batched


# TODO: Damping which loss?
class Optimizer_damping(Optimizer_batched):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_damping, self).__init__(net, N, hyperparams)

    def primal2(self, inputs, labels, index, subindex):
        max_iter = 1

        for i in range(1, max_iter + 1):
            # lagrangian, _, _, _, _ = self.augmented_lagrangian(inputs, labels)
            primal2_loss = self.primal2_loss(inputs)

            while True:
                new_params = self.cg_step(index, subindex, inputs)

                # lagrangian_new, _, _, _, _ = self.augmented_lagrangian(inputs, labels, new_params)
                primal2_loss_new = self.primal2_loss(inputs, new_params)

                # if lagrangian < lagrangian_new:
                if primal2_loss < primal2_loss_new:
                    self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                else:
                    self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                    self.restore_params(new_params)
                    break

    def cg_step(self, index, subindex, inputs):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        B, _ = self.linear_system_B(index, y_batch, J1)

        # Initial guess.
        x = torch.zeros(J2.size(1), 1, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        # L(W) = (rho/2) * norm(f(W) - r)^2
        # L_lin(d) = (rho/2) * norm(J*d - (r - f(W))^2

        # Gauss-Newton: (J2.T * J2) * x - B = 0
        # This is Gauss-Newton: new_params = self.cg_step(x, lambda p: J2.t().matmul(J2.matmul(p)), B)

        # Levenberg-Marquardt: (J2.T * J2 + M * I) * x - B = 0

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + self.hyperparams.M * p

        return self.cg_solve(x, A, B)


class Optimizer_armijo(Optimizer_batched):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_armijo, self).__init__(net, N, hyperparams)

    def lmstep(self, inputs, index, subindex, delta):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)
        B, R = self.linear_system_B(index, y_batch, J1)  # B is grad, R is residual.

        # Initial guess.
        x = torch.zeros(J2.size(1), 1, device=global_config.cfg.device)

        mu = torch.norm(R) ** delta

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + mu * p

        new_params, step = self.cg_solve(x, A, B, return_step=True)

        # Required for armijo.
        dderiv = self.hyperparams.rho * B.t().matmul(step)

        return new_params, step, B, dderiv

    def primal2(self, inputs, labels, index, subindex):
        max_iter = 1

        eps = 1e-5

        delta = self.hyperparams.delta
        eta = self.hyperparams.eta

        beta = self.hyperparams.beta
        gamma = self.hyperparams.gamma

        for i in range(1, max_iter + 1):
            loss_current = self.primal2_loss(inputs)

            new_params, step, B, dderiv = self.lmstep(inputs, index, subindex, delta)

            if torch.norm(B) <= eps:
                break

            loss_new = self.primal2_loss(inputs, new_params)

            if torch.norm(loss_new) <= eta * torch.norm(loss_current):  # TODO: Lagrangian?
                self.restore_params(new_params)
            else:
                self.armijo(inputs, step, beta, gamma, loss_current, dderiv)

    def armijo(self, inputs, step, beta, gamma, loss_current, dderiv):
        k = 1

        while True:
            sigma = (beta ** k) / beta

            if self.check_armijo(inputs, step, sigma, gamma, loss_current, dderiv):
                break

            k = k + 1

    def check_armijo(self, inputs, step, sigma, gamma, loss_current, dderiv):
        current_params = self.save_params()

        new_params = self.vec_to_params_update(sigma * step, from_numpy=False)
        self.restore_params(new_params)

        loss_new = self.primal2_loss(inputs)

        if loss_new - loss_current <= sigma * gamma * dderiv:
            return True

        self.restore_params(current_params)

        return False


class Optimizer_vanstep(Optimizer_batched):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_vanstep, self).__init__(net, N, hyperparams)

    def lmstep(self, inputs, index, subindex, delta):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)
        B, R = self.linear_system_B(index, y_batch, J1)  # B is grad, R is residual.

        # Initial guess.
        x = torch.zeros(J2.size(1), 1, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        mu = torch.norm(R) ** delta

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + mu * p

        new_params, step = self.cg_solve(x, A, B, return_step=True)

        # Required for armijo.
        dderiv = self.hyperparams.rho * B.t().matmul(step)

        return new_params, step, B, dderiv

    def primal2(self, inputs, labels, index, subindex):
        max_iter = 1

        eps = 1e-5

        delta = self.hyperparams.delta
        eta = self.hyperparams.eta

        for i in range(1, max_iter + 1):
            loss_current = self.primal2_loss(inputs)

            new_params, step, B, dderiv = self.lmstep(inputs, index, subindex, delta)

            if torch.norm(B) <= eps:
                break

            loss_new = self.primal2_loss(inputs, new_params)

            if torch.norm(loss_new) <= eta * torch.norm(loss_current):  # TODO: Lagrangian?
                self.restore_params(new_params)
            else:
                sigma = 1 / self.current_batch_iter  # TODO

                new_params = self.vec_to_params_update(sigma * step, from_numpy=False)
                self.restore_params(new_params)
