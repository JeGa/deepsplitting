import torchvision
import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import datetime

import deepsplitting.data.cifar10
import deepsplitting.data.mnist

import deepsplitting.optimizer.splitting.base as SplittingBase
import deepsplitting.utils.global_config as global_config


def imshow_grid(images, name, save):
    grid = torchvision.utils.make_grid(images.cpu(), normalize=True)
    imshow(grid, name, save)


def imshow(img, name, save):
    """
    Show img with (channel, img_size).
    """
    npimg = img.numpy()
    npimg = np.transpose(npimg, (1, 2, 0)).squeeze()

    cmap = None
    if len(npimg.shape) == 2:
        cmap = 'gray'

    plt.figure()
    plt.imshow(npimg, cmap=cmap, vmin=-1, vmax=1)
    plt.show()

    if save:
        filename = os.path.join(
            global_config.cfg.results_folder,
            global_config.cfg.results_subfolders['plots'],
            name + datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S') + '.pdf')
        plt.savefig(filename, bbox_inches='tight')


def imshow_batch_grid(loader):
    """
    Show one batch.
    """
    dataiter = iter(loader)
    images, labels = dataiter.next()

    imshow(torchvision.utils.make_grid(images))


def one_hot(x, classes, dtype=torch.double):
    """
    :param x: Torch tensor of ints of shape (N).
    :return: Torch tensor of shape (N, classes).
    """
    if x.size():
        N = x.size(0)
    else:
        N = 1

    onehot = torch.zeros(N, classes, dtype=dtype)
    onehot[list(range(N)), x] = 1

    return onehot


def cifarshow():
    trainloader, testloader, classes, training_batch_size, test_batch_size = \
        deepsplitting.data.cifar10.load_CIFAR10(8, 8)
    imshow_batch_grid(trainloader)


def is_splitting(opt):
    return isinstance(opt, SplittingBase.Optimizer)
