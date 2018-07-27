import logging
import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.base import Initializer


def total_loss(net, loader):
    loss = 0
    for inputs, label in loader:
        loss += net.loss(inputs, label)

    return loss


def train(trainloader, optimizer, epochs, params=None):
    losses = list()

    log_iter = 1

    # Only full batch.
    inputs, labels = iter(trainloader).next()

    optimizer.init(inputs, labels, Initializer.FROM_PARAMS, params)

    for epoch in range(epochs):

        optimizer.zero_grad()

        current_loss, new_loss = optimizer.step(inputs, labels)
        losses.append(current_loss)

        if epoch % log_iter == log_iter - 1:
            logging.info("{}: [{}/{}] Loss = {:.6f}".format(
                type(optimizer).__module__, epoch + 1, epochs, current_loss))

    return losses


def train_batched(trainloader, optimizer, epochs, params=None):
    total_losses = list()
    batch_losses = list()

    batch_loss_log = 10
    total_loss_log = 100

    optimizer.init(None, None, Initializer.FROM_PARAMS, params)

    # loss = total_loss(optimizer.net, trainloader)
    # total_losses.append(loss)

    # logging.info("Total: {}: [{}:{}/{}] Loss = {:.6f}".format(
    #    type(optimizer).__module__, 0, 0, epochs, loss))

    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

            optimizer.step_init(inputs, labels)

            optimizer.zero_grad()

            current_loss, new_loss = optimizer.step(inputs, labels)
            batch_losses.append(current_loss)

            if batch_loss_log != -1:
                if i % batch_loss_log == batch_loss_log - 1:
                    logging.info("Batch: {}: [{}:{}/{}] Loss = {:.6f}".format(
                        type(optimizer).__module__, epoch + 1, i + 1, epochs, current_loss))

            if total_loss_log != -1:
                if i % total_loss_log == total_loss_log - 1:
                    loss = total_loss(optimizer.net, trainloader)
                    total_losses.append(loss)

                    logging.info("Total: {}: [{}:{}/{}] Loss = {:.6f}".format(
                        type(optimizer).__module__, epoch + 1, i + 1, epochs, loss))

    return total_losses
