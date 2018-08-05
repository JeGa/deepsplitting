import torch
import logging
import math

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.splitting.base import Optimizer


class Optimizer_batched(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_batched, self).__init__(net, N, hyperparams)

    def step(self, inputs, labels):
        """
        This function does the batching. The given inputs and labels should be sampled in full batch and always be in
        the same order.
        """
        batch_size = global_config.cfg.training_batch_size
        subsample_size = int(self.hyperparams.subsample_factor * batch_size)

        indices = torch.randperm(self.N)

        max_steps = int(math.ceil(len(indices) / batch_size))

        data_loss_batchstep = list()
        lagrangian_batchstep = list()

        data_loss_startepoch, lagrangian_startepoch = self.eval(inputs, labels)

        data_loss_batchstep.append(data_loss_startepoch)
        lagrangian_batchstep.append(lagrangian_startepoch)

        lagrangian_old = lagrangian_startepoch

        for i, index in enumerate(self.batches(indices, batch_size), 1):
            subindex = torch.randperm(batch_size)[0:subsample_size]

            self.current_batch_iter = i  # For vanishing stepsize.
            self.primal2(inputs, labels, index, subindex)
            self.primal1(inputs, labels)
            self.dual(inputs)

            data_loss_new, lagrangian_new = self.eval(inputs, labels)

            data_loss_batchstep.append(data_loss_new)
            lagrangian_batchstep.append(lagrangian_new)

            logging.info("{} (Batch step [{}/{}]): Data loss = {:.6f}, Lagrangian = {:.6f}".format(
                type(self).__module__, i, max_steps, data_loss_new, lagrangian_new))

            if lagrangian_new > lagrangian_old:
                self.hyperparams.rho = self.hyperparams.rho + self.hyperparams.rho_add

            lagrangian_old = lagrangian_new

        return data_loss_batchstep, lagrangian_batchstep

    def primal2(self, inputs, labels, index, subindex):
        raise NotImplementedError()

    def linear_system_B(self, index, y_batch, J1):
        """
        This is required for GD and LM primal2 update.
        """
        R = self.v.detach()[index] - self.lam.detach()[index] / self.hyperparams.rho - y_batch
        R = torch.reshape(R, (-1, 1))

        B = J1.t().matmul(R)

        return B, R