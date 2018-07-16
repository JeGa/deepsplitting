import logging
from deepsplitting.optimizer.base import Initializer


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
            logging.info(
                "{}: [{}/{}] Loss = {:.6f}".format(type(optimizer).__module__, epoch + 1, epochs, current_loss))

    return losses
