# - vangrad
# - armijo
# - damping

# LM with CG + subsampling.

import torch
import logging
import math

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.splitting.base import Optimizer


class Optimizer_damping(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_damping, self).__init__(net, N, hyperparams)

    def step(self, inputs, labels):
        """
        This function does the batching. The given inputs and labels should be sampled in full batch and always be in
        the same order.
        """
        loss_current, lagrangian_current = self.eval(inputs, labels)

        batch_size = global_config.cfg.training_batch_size
        subsample_size = int(self.hyperparams.subsample_factor * batch_size)

        indices = torch.randperm(self.N)

        loss_batchstep = list()
        lagrangian_batchstep = list()

        max_steps = int(math.ceil(len(indices) / batch_size))
        i = 0

        for index in self.batches(indices, batch_size):
            i += 1

            subindex = torch.randperm(batch_size)[0:subsample_size]

            self.primal2(inputs, labels, index, subindex)
            self.primal1(inputs, labels)
            self.dual(inputs)

            data_loss_batch, lagrangian_batch = self.eval(inputs, labels)

            loss_batchstep.append(data_loss_batch)
            lagrangian_batchstep.append(lagrangian_batch)

            logging.info("{} (Batch step [{}/{}]): Data loss = {:.6f}, Lagrangian = {:.6f}".format(
                type(self).__module__, i, max_steps, data_loss_batch, lagrangian_batch))

            loss_new, lagrangian_new = self.eval(inputs, labels)

            if lagrangian_new > lagrangian_current:
                self.hyperparams.rho = self.hyperparams.rho + self.hyperparams.rho_add

        return loss_current, loss_new, lagrangian_current, lagrangian_new, loss_batchstep, lagrangian_batchstep

    def subsampled_jacobians(self, index, subindex, inputs):
        y_batch = self.forward(inputs[index], requires_grad=True)

        J1 = self.jacobian_torch(y_batch).detach()

        y_subsampled = y_batch[subindex]

        c = y_subsampled.size(1)
        subsampled_indices = [item for j in subindex for item in range(j * c, j * c + c)]

        J2 = J1[subsampled_indices, :]

        return J1, J2, y_batch.detach(), y_subsampled.detach()

    def linear_system_B(self, index, y_batch, J1):
        R = self.v.detach()[index] - self.lam.detach()[index] / self.hyperparams.rho - y_batch
        R = torch.reshape(R, (-1, 1))

        B = J1.t().matmul(R)

        return B

    def cg_step(self, index, subindex, inputs):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        B = self.linear_system_B(index, y_batch, J1)

        # Initial guess.
        x = torch.zeros(J2.size(1), 1, device=global_config.cfg.device)

        # Gauss-Newton: (J2.T * J2) * x - B = 0
        # This is Gauss-Newton: new_params = self.cg_step(x, lambda p: J2.t().matmul(J2.matmul(p)), B)

        # Levenberg-Marquardt: (J2.T * J2 + M * I) * x - B = 0

        def A(p):
            return J2.t().matmul(J2.matmul(p)) + self.hyperparams.M * p

        return self.cg_solve(x, A, B)

    def gd_step(self, index, subindex, inputs):
        J1, J2, y_batch, _ = self.subsampled_jacobians(index, subindex, inputs)

        B = self.linear_system_B(index, y_batch, J1)

        return self.vec_to_params_update(self.hyperparams.rho * 1e-3 * B, from_numpy=False)

    def primal2(self, inputs, labels, index, subindex):
        max_iter = 1

        for i in range(1, max_iter + 1):
            lagrangian, _, _, _, _ = self.augmented_lagrangian(inputs, labels)

            while True:
                new_params = self.cg_step(index, subindex, inputs)

                lagrangian_new, _, _, _, _ = self.augmented_lagrangian(inputs, labels, new_params)

                if lagrangian < lagrangian_new:
                    self.hyperparams.M = self.hyperparams.M * self.hyperparams.factor
                else:
                    self.hyperparams.M = self.hyperparams.M / self.hyperparams.factor
                    self.restore_params(new_params)
                    break

    def primal2_gd(self, inputs, index, subindex):
        new_params = self.gd_step(index, subindex, inputs)
        self.restore_params(new_params)

    def cg_solve(self, x, A, B):
        """
        Solve for: Ax = B with linear mapping A.

        The argument A should be a function. It is often better to calculate e.g. Ax = A1*A2*x using matrix vector
        products instead of calculating A directly.
        """
        r = B - A(x)
        p = r

        for i in range(self.hyperparams.cg_iter):
            alpha = (r.t().matmul(r)) / (p.t().matmul(A(p)))

            x = x + alpha * p
            r_new = r - alpha * A(p)

            beta = (r_new.t().matmul(r_new)) / (r.t().matmul(r))
            p = r_new + beta * p

            r = r_new

        return self.vec_to_params_update(x, from_numpy=False)
