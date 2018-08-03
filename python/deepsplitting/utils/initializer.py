import torch
from torch.nn import functional as F

import deepsplitting.data.spirals
import deepsplitting.data.cifar10
import deepsplitting.data.mnist
import deepsplitting.losses.mse
import deepsplitting.networks.simple
import deepsplitting.utils
import deepsplitting.utils.global_config as global_config


def get_activation(activation_type):
    if activation_type == 'relu':
        activation = F.relu
    elif activation_type == 'sigmoid':
        activation = F.sigmoid
    else:
        raise ValueError('Unsupported activation type.')

    return activation


def get_loss(loss_type):
    if loss_type == 'ls':
        loss = deepsplitting.losses.mse.WeightedMSELoss()
    elif loss_type == 'nll':
        loss = torch.nn.CrossEntropyLoss(size_average=False)
    else:
        raise ValueError("Unsupported loss type.")

    return loss


def mnist_tf(loss_type):
    if loss_type == 'ls':
        def target_transform(target):
            return deepsplitting.utils.misc.one_hot(target, 10).squeeze()

        tf = target_transform
    elif loss_type == 'nll':
        tf = None
    else:
        raise ValueError("Unsupported loss type.")

    return tf


def ff_spirals_tf(loss_type):
    if loss_type == 'ls':
        tf = None
    elif loss_type == 'nll':
        def target_transform(target):
            return target.argmax()

        tf = target_transform
    else:
        raise ValueError("Unsupported loss type.")

    return tf


def cnn_CIFAR10_tf(loss_type):
    if loss_type == 'ls':
        def target_transform(target):
            return deepsplitting.utils.misc.one_hot(torch.tensor(target), 10).squeeze()

        tf = target_transform
    elif loss_type == 'nll':
        tf = None
    else:
        raise ValueError("Unsupported loss type.")

    return tf


def ff_mnist_vectorized(loss_type, activation_type):
    tf = mnist_tf(loss_type)
    loss = get_loss(loss_type)

    layers = [784, 10, 10]
    activation = get_activation(activation_type)

    net = deepsplitting.networks.simple.SimpleFFNet(layers, activation, loss)
    trainloader, testloader, training_batch_size, test_batch_size = \
        deepsplitting.data.mnist.load_MNIST_vectorized(-1, 8, target_transform=tf)

    return net, trainloader, testloader, training_batch_size, test_batch_size


def ff_spirals(loss_type, activation_type):
    tf = ff_spirals_tf(loss_type)
    loss = get_loss(loss_type)

    layers = [2, 12, 12, 12, 2]

    activation = get_activation(activation_type)

    net = deepsplitting.networks.simple.SimpleFFNet(layers, activation, loss)

    net.to(global_config.cfg.device)

    trainloader, training_batch_size = deepsplitting.data.spirals.load_spirals(training_samples=-1, target_transform=tf)

    return net, trainloader, training_batch_size


def cnn_cifar10(loss_type, activation_type):
    tf = cnn_CIFAR10_tf(loss_type)
    loss = get_loss(loss_type)

    activation = get_activation(activation_type)

    net = deepsplitting.networks.simple.SimpleConvNet_cifar(activation, loss)

    net.to(global_config.cfg.device)

    # trainloader, _, classes, training_batch_size, _ = deepsplitting.data.cifar10.load_CIFAR10_batched(
    #    global_config.cfg.training_batch_size, 15, target_transform=tf, folder='/work/ganslose/data')

    trainloader, _, classes, training_batch_size, _ = deepsplitting.data.cifar10.load_CIFAR10(training_samples=10,
                                                                                              target_transform=tf)

    return net, trainloader, training_batch_size, classes


def cnn_mnist(loss_type, activation_type):
    tf = mnist_tf(loss_type)
    loss = get_loss(loss_type)

    activation = get_activation(activation_type)

    net = deepsplitting.networks.simple.SimpleConvNet_mnist(activation, loss)

    net.to(global_config.cfg.device)

    trainloader, testloader, training_batch_size, test_batch_size, classes = deepsplitting.data.mnist.load_MNIST(
        training_samples=global_config.cfg.training_samples, target_transform=tf, fullbatch=True,
        training_batch_size=global_config.cfg.training_batch_size, test_batch_size=1)

    return net, trainloader, training_batch_size, classes
