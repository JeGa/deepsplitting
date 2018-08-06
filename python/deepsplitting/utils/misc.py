import torchvision
import torch
import os.path
import itertools
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime

import deepsplitting.data.cifar10
import deepsplitting.data.mnist

import deepsplitting.utils.global_config as global_cfg
import deepsplitting.optimizer.splitting.base as SplittingBase


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


def plot_loss_curve(losses, title=''):
    plt.figure()

    plt.title(title)

    plt.plot(losses, linewidth=1.0)
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.show()


def plot_summary(summary, timer, optimizer, filename, folder):
    marker = itertools.cycle(('s', 'D', '.', 'o', '^', 'v', '*', '8', 'x'))
    marker_factor = 0.04
    marker_min = 4

    plt.figure()
    plt.title("Loss: {}, activation: {}".format(global_cfg.cfg.loss_type, global_cfg.cfg.activation_type))

    N = len(next(iter(next(iter(summary.values())).values())))
    every = max(int(marker_factor * N), marker_min)
    hyperparams_y = -0.1

    for optimizer_key, all_losses in summary.items():
        for loss_key, losses in all_losses.items():
            plt.plot(losses, label=optimizer_key + ' ' + loss_key + ' ' + time_str(optimizer_key, timer) + 's',
                     linewidth=1.0, marker=next(marker), markevery=every, markerfacecolor='none')

        plt.text(0, hyperparams_y, optimizer_key + ': ' + str(optimizer[optimizer_key].hyperparams.csv_format()),
                 transform=plt.gcf().transFigure)
        hyperparams_y -= -0.03

        every += 1

    plt.legend()
    plt.xlabel('Iteration')
    plt.ylabel('Objective')

    plt.savefig(os.path.join(folder, filename + '.pdf'), bbox_inches='tight')


def cifarshow():
    trainloader, testloader, classes, training_batch_size, test_batch_size = \
        deepsplitting.data.cifar10.load_CIFAR10(8, 8)
    show(trainloader)


def is_splitting(opt):
    return isinstance(opt, SplittingBase.Optimizer)


def save_csv(name, data, params):
    folder = os.path.join(global_cfg.cfg.results_folder, global_cfg.cfg.results_subfolders['data'])

    with open(os.path.join(folder, name), 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(['GLOBAL_CONFIG'] + global_cfg.cfg.csv_format())

        if params is not None:
            writer.writerow(['PARAMETERS'] + params.csv_format())

        writer.writerows(data)


def time_str(key, timer):
    return "{:.6f}".format(timer.times[key]) if key in timer.times else ''


def save_summary(optimizer, summary, timer):
    for optimizer_key, all_losses in summary.items():
        timerstr = time_str(optimizer_key, timer)

        all_losses = [[key] + value for key, value in all_losses.items()]

        save_csv("{}_{}_{}.csv".format(optimizer_key, datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S'), timerstr),
                 all_losses, optimizer[optimizer_key].hyperparams)


def mkdir_ifnot(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def make_results_folder(cfg):
    mkdir_ifnot(cfg.results_folder)

    for key, f in cfg.results_subfolders.items():
        mkdir_ifnot(os.path.join(cfg.results_folder, f))
