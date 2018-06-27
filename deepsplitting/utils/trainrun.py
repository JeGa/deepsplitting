import logging


def train(net, trainloader, optimizer, epochs):
    losses = list()

    for epoch in range(epochs):
        for i, data in enumerate(trainloader):
            inputs, labels = data

            net.zero_grad()

            loss = optimizer.step(inputs, labels)
            losses.append(loss)

        if epoch % 10 == 9:
            logging.info('[%d/%d] loss: %.3f' % (epoch + 1, epochs, loss))

    return losses
