import torch
from torch.nn import functional as F

import deepsplitting.data.spirals
import deepsplitting.losses.mse
import deepsplitting.networks.simple
import deepsplitting.utils


def ff_mnist_vectorized(loss_type):
    if loss_type == 'ls':
        def target_transform(target):
            return deepsplitting.utils.misc.one_hot(target, 10).squeeze()

        tf = target_transform
        loss = torch.nn.MSELoss()
    elif loss_type == 'nll':
        tf = None
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type.")

    layers = [784, 10, 10]
    activation = F.relu

    net = deepsplitting.networks.simple.SimpleFFNet(layers, activation, loss).double()
    trainloader, testloader, training_batch_size, test_batch_size = \
        deepsplitting.data.mnist.load_MNIST_vectorized(8, 8, target_transform=tf)

    return net, trainloader, testloader, training_batch_size, test_batch_size


def ff_spirals(loss_type):
    if loss_type == 'ls':
        tf = None
        loss = deepsplitting.losses.mse.WeightedMSELoss()
    elif loss_type == 'nll':
        def target_transform(target):
            return target.argmax()

        tf = target_transform
        loss = torch.nn.CrossEntropyLoss()
    else:
        raise ValueError("Unsupported loss type.")

    layers = [2, 12, 2]
    activation = F.relu

    net = deepsplitting.networks.simple.SimpleFFNet(layers, activation, loss).double()
    trainloader, training_batch_size = deepsplitting.data.spirals.load_spirals(target_transform=tf)

    return net, trainloader, training_batch_size
