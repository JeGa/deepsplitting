import torchvision
import torch

import matplotlib.pyplot as plt
import numpy as np

import deepsplitting.data.cifar10
import deepsplitting.data.mnist


def imshow(img, factor=1 / (2 + 0.5)):
    img = img * factor
    npimg = img.numpy()

    plt.figure()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def show(loader):
    """
    Show one batch.
    """
    dataiter = iter(loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))


def one_hot(x, classes):
    """
    :param x: Torch tensor of ints of shape (N).
    :return: Torch tensor of floats of shape (N, classes).
    """
    if x.size():
        N = x.size(0)
    else:
        N = 1

    onehot = torch.zeros((N, classes), dtype=torch.float64)
    onehot[list(range(N)), x] = 1

    return onehot


def plot_loss_curve(losses, title=''):
    plt.figure()

    plt.title(title)

    plt.plot(losses)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()


def plot_summary(summary, title=''):
    plt.figure()

    plt.title(title)

    for key, losses in summary.items():
        plt.plot(losses, label=key)

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()


def cifarshow():
    trainloader, testloader, classes, training_batch_size, test_batch_size = \
        deepsplitting.data.cifar10.load_CIFAR10(8, 8)
    show(trainloader)
