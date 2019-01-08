import numpy as np
import torch
import logging

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.base import BaseOptimizer
from deepsplitting.optimizer.misc import prox_cross_entropy
from deepsplitting.optimizer.base import Initializer


class Optimizer(BaseOptimizer):
    def __init__(self, net, N, hyperparams):
        super(Optimizer, self).__init__(net, hyperparams)

        if type(self.net.criterion) is torch.nn.CrossEntropyLoss:
            self.primal1_loss = primal1_nll
        elif isinstance(self.net.criterion, torch.nn.MSELoss):
            self.primal1_loss = primal1_ls
        else:
            raise AttributeError('Loss not supported: No primal1 update function.')

        self.N = N

        self.lam = None
        self.v = None

        self.init_variables(Initializer.RANDN)

    def init_variables(self, initializer):
        self.lam = torch.zeros(self.N, self.net.output_dim,
                               dtype=global_config.cfg.datatype,
                               device=global_config.cfg.device)

        if initializer is Initializer.DEBUG:
            self.v = torch.zeros(self.N, self.net.output_dim,
                                 dtype=global_config.cfg.datatype,
                                 device=global_config.cfg.device)
        else:
            if global_config.cfg.seed != -1:
                torch.manual_seed(global_config.cfg.seed)

            self.v = 0.1 * torch.randn(self.N, self.net.output_dim,
                                       dtype=global_config.cfg.datatype,
                                       device=global_config.cfg.device)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

        self.init_variables(initializer)

    def step_init(self, inputs, labels):
        raise NotImplementedError()

    def eval(self, inputs, labels, print=False):
        lagrangian, data_loss, lagrangian_data_loss, constraint_norm, _ = self.augmented_lagrangian(inputs, labels)

        if print:
            logging.info("Data loss = {:.8f}, Lagrangian data loss = {:.8f}, "
                         "constraint norm = {:.8f}, Lagrangian = {:.8f}".format(data_loss, lagrangian_data_loss,
                                                                                constraint_norm, lagrangian))

        return data_loss, lagrangian

    def augmented_lagrangian(self, inputs, labels, params=None):
        saved_params = self.load(params)

        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor)

        data_loss = self.net.criterion(y, labels)

        # TODO: This compares the losses.
        # data_loss = 0.5 * torch.pow((y - labels), 2).sum()
        # print(data_loss, data_loss_tmp)

        lagrangian_data_loss = self.net.criterion(self.v, labels)
        constraint = y - self.v
        constraint_norm = torch.pow(constraint, 2).sum()

        lagrangian = lagrangian_data_loss + torch.mul(self.lam, constraint).sum() + (
                self.hyperparams.rho / 2) * constraint_norm

        self.restore(saved_params)

        return lagrangian.item(), data_loss.item(), lagrangian_data_loss.item(), constraint_norm.item(), y

    def primal2_loss(self, inputs, params=None):
        saved_params = self.load(params)

        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor)

        rho = self.hyperparams.rho
        loss = (rho / 2) * torch.pow(y - self.v + (1 / rho) * self.lam, 2).sum()

        self.restore(saved_params)

        return loss

    def primal1(self, inputs, labels):
        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor).detach()

        self.v = self.primal1_loss(y, self.lam.detach(), labels, self.hyperparams.rho, self.net.criterion)

    def dual(self, inputs):
        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor).detach()

        self.lam = self.lam.detach() + self.hyperparams.rho * (y - self.v)


def primal1_ls(y, lam, y_train, rho, loss):
    C = 2 * loss.C
    if loss.size_average is True:
        C = C * 1 / y_train.size(0)

    return (C * y_train + rho * y + lam) / (C + rho)


def primal1_nll(y, lam, y_train, rho, loss):
    """
    Return the solution of the prox operator of the nll loss.

    :param y: f(u^k+1)
    :param lam: lambda^k
    :param y_train: Ground truth labels.
    :param rho: Prox factor.
    :param loss: pytorch nll loss function.
    :return New primal1 variable (z^k+1).
    """

    # Primal1 problem:
    #   z^k+1 = argmin_z L(z;y) + (rho/2) * norm(z - R)^2 = prox_L(R)
    # with R = f(u^k+1) + (1/rho) * lambda^k, L(z) = -log(softmax(z)) and y the ground truth label.

    z_new = torch.zeros(y.size(), dtype=global_config.cfg.datatype, device=global_config.cfg.device)

    R = y + lam / rho

    for i in range(y_train.size(0)):
        with torch.no_grad():
            y_train_i = y_train[i].item()
            Ri = R[i].cpu().numpy()

            z_new_i = prox_cross_entropy(Ri, 1 / rho, y_train_i)
            z_new[i] = torch.tensor(z_new_i, dtype=global_config.cfg.datatype, device=global_config.cfg.device)

    return z_new
