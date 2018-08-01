import torchvision
import torch
import os.path
import itertools
import matplotlib.pyplot as plt
import numpy as np

import deepsplitting.data.cifar10
import deepsplitting.data.mnist
import deepsplitting.utils.global_config as global_config


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

    onehot = torch.zeros(N, classes, dtype=torch.float)
    onehot[list(range(N)), x] = 1

    return onehot


def plot_loss_curve(losses, title=''):
    plt.figure()

    plt.title(title)

    plt.plot(losses, linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()


def plot_summary(summary, timer, name, title='', folder='results'):
    marker = itertools.cycle(('s', 'D', '.', 'o', '^', 'v', '*', '8', 'x'))

    plt.figure()

    plt.title(title)

    every = 4

    for key, losses in summary.items():
        plt.plot(losses,
                 label="{} {:.6f}s".format(key, timer.times[key]) if key in timer.times else "{}".format(key),
                 linewidth=1.0, marker=next(marker), markevery=every, markerfacecolor='none')

        every += 1

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.savefig(os.path.join(folder, name + '.pdf'), bbox_inches='tight')


def cifarshow():
    trainloader, testloader, classes, training_batch_size, test_batch_size = \
        deepsplitting.data.cifar10.load_CIFAR10(8, 8)
    show(trainloader)


def is_llc(opt):
    return type(opt) is deepsplitting.optimizer.llc.Optimizer
