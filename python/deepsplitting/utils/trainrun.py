import logging

import deepsplitting.utils.global_config as global_config
import deepsplitting.utils.global_progressbar as pb

from deepsplitting.optimizer.base import Initializer


def total_loss(net, loader):
    loss = 0
    for inputs, labels in loader:
        inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)
        loss += net.loss(inputs, labels).item()

    return loss


def train_splitting(trainloader, optimizer, params=None):
    data_loss = list()
    lagrangian = list()

    log_iter = 1

    # Full batch. Batching is done by the optimizer.
    inputs, labels = iter(trainloader).next()
    inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

    pb.init(global_config.cfg.epochs, global_config.cfg.training_batch_size, inputs.size(0),
            dict(dataloss='Data loss', lagrangian='Lagrangian'))

    optimizer.init(inputs, labels, Initializer.FROM_PARAMS, params)

    for epoch in range(global_config.cfg.epochs):
        optimizer.zero_grad()

        data_loss_batchstep, lagrangian_batchstep = optimizer.step(inputs, labels)

        data_loss += data_loss_batchstep
        lagrangian += lagrangian_batchstep

        if epoch % log_iter == log_iter - 1:
            logging.info("{}: [{}/{}]".format(type(optimizer).__module__, epoch + 1, global_config.cfg.epochs))

    pb.bar.finish()

    return data_loss, lagrangian


def train_LM_GD(trainloader, optimizer, params=None):
    data_loss = list()

    log_iter = 1

    # Full batch. Batching is done by the optimizer.
    inputs, labels = iter(trainloader).next()
    inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

    pb.init(global_config.cfg.epochs, global_config.cfg.training_batch_size, inputs.size(0),
            dict(dataloss='Data loss'))

    optimizer.init(inputs, labels, Initializer.FROM_PARAMS, params)

    for epoch in range(global_config.cfg.epochs):
        optimizer.zero_grad()

        data_loss_batchstep = optimizer.step(inputs, labels)

        data_loss += data_loss_batchstep

        if epoch % log_iter == log_iter - 1:
            logging.info("{}: [{}/{}]".format(type(optimizer).__module__, epoch + 1, global_config.cfg.epochs))

    pb.bar.finish()

    return data_loss
