import datetime
import itertools
import os.path

import yaml
from matplotlib import pyplot as plt

from deepsplitting.utils import global_config as global_cfg


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


def load_yaml(name):
    folder = os.path.join(global_cfg.cfg.results_folder, global_cfg.cfg.results_subfolders['data'])

    with open(os.path.join(folder, name), 'r', newline='') as f:
        yaml_dict = yaml.load(f)


def save_yaml(name, time, data, params):
    folder = os.path.join(global_cfg.cfg.results_folder, global_cfg.cfg.results_subfolders['data'])

    filepath = os.path.join(folder, name)

    with open(filepath, 'w', newline='') as f:
        to_write = []

        if params is not None:
            params = {'Parameters': params.__dict__}

        cfgdict = {'Global_config': global_cfg.cfg.__dict__}
        timedict = {'Time': time}
        datadict = {'Data': data}

        if params is not None:
            to_write.append(params)
        to_write.append(cfgdict)
        to_write.append(timedict)
        to_write.append(datadict)

        yaml.dump(to_write, f, default_flow_style=False)

        print("Saved results as {}.".format(filepath))


def time_str(key, timer):
    return "{:.6f}".format(timer.times[key]) if key in timer.times else ''


def save_summary(optimizer, summary, timer):
    for optimizer_key, all_losses in summary.items():
        timerstr = time_str(optimizer_key, timer)

        filename = "{}_{}.yml".format(optimizer_key, datetime.datetime.now().strftime('%d-%m-%y_%H:%M:%S'))

        save_yaml(filename, timerstr, all_losses, optimizer[optimizer_key].hyperparams)


def mkdir_ifnot(dir):
    if not os.path.exists(dir):
        os.mkdir(dir)


def make_results_folder(cfg):
    mkdir_ifnot(cfg.results_folder)

    for key, f in cfg.results_subfolders.items():
        mkdir_ifnot(os.path.join(cfg.results_folder, f))