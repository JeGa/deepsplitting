import logging

import matplotlib.pyplot as plt

import deepsplitting.optimizer.gradient_descent as GD
import deepsplitting.optimizer.gradient_descent_armijo as GDA
import deepsplitting.optimizer.llc as LLC
import deepsplitting.optimizer.prox_descent as ProxDescent
from deepsplitting.utils.initializer import ff_spirals
from deepsplitting.utils.testrun import test_ls
from deepsplitting.utils.trainrun import train


def main():
    logging.basicConfig(level=logging.INFO)

    loss_type = 'ls'

    net, trainloader, training_batch_size = ff_spirals(loss_type)

    optimizer = {'LLC': LLC.Optimizer(net, N=training_batch_size),  # Not with nll
                 'ProxDescent': ProxDescent.Optimizer(net),  # not with nll
                 'GDA': GDA.Optimizer(net),
                 'GD': GD.Optimizer(net)}

    losses = train(net, trainloader, optimizer['ProxDescent'], 80)

    plt.figure()
    plt.plot(losses)
    plt.show()

    if loss_type == 'ls':
        test_ls(net, trainloader, 2)


if __name__ == '__main__':
    main()
