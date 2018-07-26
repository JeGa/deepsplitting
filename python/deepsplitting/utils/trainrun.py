import logging

from deepsplitting.optimizer.base import Initializer


def total_loss(net, loader):
    loss = 0
    for inputs, label in loader:
        print(loss)
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
    losses = list()

    log_iter = 100

    optimizer.init(None, None, Initializer.FROM_PARAMS, params)

    #loss = total_loss(optimizer.net, trainloader)
    #print('dddddddddddddddddd')

    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data
            optimizer.step_init(inputs, labels)

            optimizer.zero_grad()

            current_loss, new_loss = optimizer.step(inputs, labels)
            losses.append(current_loss)

            if i % log_iter == log_iter - 1:
                logging.info("{}: [{}:{}/{}] Loss = {:.6f}".format(
                    type(optimizer).__module__, epoch + 1, i + 1, epochs, current_loss))

        #loss = total_loss(optimizer.net, trainloader)
        #logging.info("[{}/{}] Loss = {:.6f}".format(epoch + 1, epochs, loss))

    return losses
