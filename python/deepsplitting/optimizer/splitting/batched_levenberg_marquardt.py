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

            # self.primal2_levmarq_batched_Mfac(inputs, labels, index, subindex)

            self.primal1(inputs, labels)
            self.dual(inputs)

            data_loss_batch, lagrangian_batch = self.eval(inputs, labels)

            loss_batchstep.append(data_loss_batch.item())
            lagrangian_batchstep.append(lagrangian_batch.item())

            logging.info("{} (Batch step [{}/{}]): Data loss = {:.6f}, Lagrangian = {:.6f}".format(
                type(self).__module__, i, max_steps, data_loss_batch, lagrangian_batch))

        loss_new, lagrangian_new = self.eval(inputs, labels)

        if lagrangian_new > lagrangian_current:
            self.hyperparams.rho = self.hyperparams.rho + self.hyperparams.rho_add

        return (loss_current.item(), loss_new.item(),
                lagrangian_current.item(), lagrangian_new.item(),
                loss_batchstep, lagrangian_batchstep)
