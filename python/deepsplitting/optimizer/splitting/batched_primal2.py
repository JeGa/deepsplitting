import torch
import logging

import deepsplitting.utils.global_config as global_config
import deepsplitting.utils.global_progressbar as gp

from deepsplitting.optimizer.splitting.base import Optimizer


class Optimizer_batched(Optimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer_batched, self).__init__(net, N, hyperparams)

        self.iteration = 1

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer_batched, self).init(inputs, labels, initializer, parameters)

        self.iteration = 1

    def step(self, inputs, labels):
        """
        This function does the batching. The given inputs and labels should be sampled in full batch and always be in
        the same order.
        """
        batch_size = global_config.cfg.training_batch_size
        indices, subsample_size, max_steps = self.fullbatch_subindex_init(batch_size,
                                                                          self.hyperparams.subsample_factor,
                                                                          self.N)

        data_loss_batchstep = list()
        lagrangian_batchstep = list()

        data_loss_startepoch, lagrangian_startepoch = self.eval(inputs, labels)

        data_loss_batchstep.append(data_loss_startepoch)
        lagrangian_batchstep.append(lagrangian_startepoch)

        lagrangian_old = lagrangian_startepoch

        for i, index in enumerate(self.batches(indices, batch_size), 1):
            subindex = self.rand_subindex(batch_size, subsample_size)

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

            self.iteration += 1

            gp.bar.next_batch(dict(dataloss=data_loss_new, lagrangian=lagrangian_new))

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
