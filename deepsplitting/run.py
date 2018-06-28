import matplotlib.pyplot as plt
import logging
import torch.nn as nn
import torch.nn.functional as F

import deepsplitting.data
import deepsplitting.networks.simple
import deepsplitting.utils.misc
import deepsplitting.optimizer.llc as LLC
import deepsplitting.optimizer.gradient_descent_armijo as GDA
import deepsplitting.optimizer.gradient_descent as GD

from deepsplitting.utils.trainrun import train
from deepsplitting.utils.testrun import test_nll, test_ls


def initialize(loss_type):
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
           deepsplitting.data.load_MNIST_vectorized(16, 8, ttransform=tf)


def main():
    logging.basicConfig(level=logging.INFO)

    net, trainloader, testloader, training_batch_size, _ = initialize('ls')

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
