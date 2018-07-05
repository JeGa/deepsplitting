import matplotlib.pyplot as plt
import logging
import torch.nn as nn
import torch.nn.functional as F

import deepsplitting.utils.data
import deepsplitting.networks.simple
import deepsplitting.utils.misc

import deepsplitting.optimizer.llc as LLC
import deepsplitting.optimizer.gradient_descent_armijo as GDA
import deepsplitting.optimizer.gradient_descent as GD

from deepsplitting.utils.trainrun import train
from deepsplitting.utils.testrun import test_ls


def mnist(loss_type):
    if loss_type == 'ls':
        def target_transform(target):
            return deepsplitting.utils.misc.one_hot(target, 10).squeeze()

        tf = target_transform
        loss = nn.MSELoss()
    elif loss_type == 'nll':
        tf = None
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type.")

    return (deepsplitting.networks.simple.SimpleNet(F.relu, loss),) + \
           deepsplitting.utils.data.load_MNIST_vectorized(16, 8, ttransform=tf)


def spirals(loss_type):
    if loss_type == 'ls':
        tf = None
        loss = nn.MSELoss()
    elif loss_type == 'nll':
        def target_transform(target):
            return target.argmax()

        tf = target_transform
        loss = nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type.")

    return (deepsplitting.networks.simple.SimpleNet(F.relu, loss),) + \
           deepsplitting.utils.data.load_spirals(ttransform=tf)


def cifarshow():
    trainloader, testloader, classes, training_batch_size, test_batch_size = deepsplitting.utils.data.load_CIFAR10(8, 8)
    deepsplitting.utils.misc.show(trainloader)


def main():
    logging.basicConfig(level=logging.INFO)

    # net, trainloader, testloader, training_batch_size, _ = mnist('ls')

    net, trainloader, training_batch_size = spirals('ls')

    deepsplitting.utils.misc.show(trainloader)

    optimizer = {'LLC': LLC.Optimizer(net, N=training_batch_size),
                 'GDA': GDA.Optimizer(net),
                 'GD': GD.Optimizer(net)}

    losses = train(net, trainloader, optimizer['LLC'], 20)

    plt.figure()
    plt.plot(losses)
    plt.show()

    test_ls(net, trainloader)


if __name__ == '__main__':
    main()
