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
    data_loss = []
    lagrangian = []
    correctly_classified = []

    log_iter = 1

    # Full batch. Batching is done by the optimizer.
    inputs, labels = iter(trainloader).next()
    inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

    # For autoencoder:
    # labels = inputs.view(inputs.size(0), -1)  # TODO: labels

    pb.init(global_config.cfg.epochs, global_config.cfg.training_batch_size, inputs.size(0),
            dict(dataloss='Data loss', lagrangian='Lagrangian'))

    optimizer.init(inputs, labels, Initializer.FROM_PARAMS, params)

    for epoch in range(global_config.cfg.epochs):
        optimizer.zero_grad()

        data_loss_batchstep, lagrangian_batchstep, correct = optimizer.step(inputs, labels)

        data_loss += data_loss_batchstep
        lagrangian += lagrangian_batchstep
        correctly_classified += correct

        if epoch % log_iter == log_iter - 1:
            logging.info("{}: [{}/{}]".format(type(optimizer).__module__, epoch + 1, global_config.cfg.epochs))

    pb.bar.finish()

    return data_loss, lagrangian, correctly_classified


import deepsplitting.utils.misc as misc


def aetest(net, inputs, n=8 * 4):
    out = net(inputs[0:n]).view(inputs[0:n].size())

    misc.imshow_grid(inputs[0:n], 'input', save=True)
    misc.imshow_grid(out[0:n].detach(), 'output', save=True)


def train_LM_GD(trainloader, optimizer, params=None):
    data_loss = []
    correctly_classified = []

    log_iter = 1

    # Full batch. Batching is done by the optimizer.
    inputs, labels = iter(trainloader).next()
    inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

    # For autoencoder:
    # labels = inputs.view(inputs.size(0), -1)  # TODO: labels

    pb.init(global_config.cfg.epochs, global_config.cfg.training_batch_size, inputs.size(0),
            dict(dataloss='Data loss'))

    optimizer.init(inputs, labels, Initializer.FROM_PARAMS, params)

    for epoch in range(global_config.cfg.epochs):
        optimizer.zero_grad()

        data_loss_batchstep, correct = optimizer.step(inputs, labels)

        data_loss += data_loss_batchstep
        correctly_classified += correct

        if epoch % log_iter == log_iter - 1:
            logging.info("{}: [{}/{}]".format(type(optimizer).__module__, epoch + 1, global_config.cfg.epochs))

    pb.bar.finish()

    # aetest(optimizer.net, inputs)

    return data_loss, correctly_classified
