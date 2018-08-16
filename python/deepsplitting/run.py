import matplotlib

import deepsplitting.utils.evaluate

matplotlib.use('Agg')

import logging
from collections import namedtuple

import progressbar

progressbar.streams.wrap_stderr()

import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.timing as timing
import deepsplitting.utils.misc
import deepsplitting.config as config_file
import deepsplitting.utils.global_config as global_config

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM
import deepsplitting.optimizer.lm.batched_levenberg_marquardt as bLM

import deepsplitting.optimizer.splitting.batched_gradient_descent as sbGD
import deepsplitting.optimizer.gd.gradient_descent as GD


def main():
    if global_config.cfg.logging != -1:
        logging.basicConfig(level=global_config.cfg.logging)

    deepsplitting.utils.evaluate.make_results_folder(global_config.cfg)

    net, trainloader, training_batch_size, classes = initializer.cnn_mnist(global_config.cfg.loss_type,
                                                                           global_config.cfg.activation_type)

    if global_config.cfg.loss_type == 'ls':
        optimizer_params = config_file.optimizer_params_ls
    elif global_config.cfg.loss_type == 'nll':
        optimizer_params = config_file.optimizer_params_nll
    else:
        raise ValueError("Unsupported loss type.")

    OptimizerEntry = namedtuple('OptimizerEntry', ['on', 'key', 'optimizer'])

    optimizer = [
        OptimizerEntry(
            False, 'sbLM_damping',
            sbLM.Optimizer_damping(net, N=training_batch_size, hyperparams=optimizer_params['sbLM_damping'])),
        OptimizerEntry(
            False, 'sbLM_armijo',
            sbLM.Optimizer_armijo(net, N=training_batch_size, hyperparams=optimizer_params['sbLM_armijo'])),
        OptimizerEntry(
            False, 'sbLM_vanstep',
            sbLM.Optimizer_vanstep(net, N=training_batch_size, hyperparams=optimizer_params['sbLM_vanstep'])),

        OptimizerEntry(
            False, 'sbGD_fix',
            sbGD.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['sbGD_fix'])),
        OptimizerEntry(
            False, 'sbGD_vanstep',
            sbGD.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['sbGD_vanstep'])),

        OptimizerEntry(
            True, 'bLM_damping',
            bLM.Optimizer_damping(net, N=training_batch_size, hyperparams=optimizer_params['bLM_damping'])),
        OptimizerEntry(
            False, 'bLM_armijo',
            bLM.Optimizer_armijo(net, N=training_batch_size, hyperparams=optimizer_params['bLM_armijo'])),
        OptimizerEntry(
            False, 'bLM_vanstep',
            bLM.Optimizer_vanstep(net, N=training_batch_size, hyperparams=optimizer_params['bLM_vanstep'])),

        OptimizerEntry(
            False, 'bGD_fix',
            GD.Optimizer_batched(net, hyperparams=optimizer_params['bGD_fix'])),
        OptimizerEntry(
            False, 'bGD_vanstep',
            GD.Optimizer_batched(net, hyperparams=optimizer_params['bGD_vanstep']))

        # Other stuff.
        # 'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        # 'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),
        # 'ProxProp': ProxProp.Optimizer(net, hyperparams=optimizer_params['ProxProp'])
    ]

    optimizer = {opt.key: opt.optimizer for opt in optimizer if opt.on}

    # trainrun.train(trainloader, )

    # if params.loss_type == 'ls':
    #    optimizer['LM'] = LM.Optimizer(net, hyperparams=optimizer_params['LM'])

    # To have the same network parameter initialization for all nets.
    net_init_parameters = next(iter(optimizer.values())).save_params()

    train_all(optimizer, trainloader, net_init_parameters, classes)


def train_splitting(opt, key, trainloader, net_init_parameters, summary):
    data_loss, lagrangian = trainrun.train_splitting(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
        'lagrangian': lagrangian
    }

    summary[key] = results


def train_lm(opt, key, trainloader, net_init_parameters, summary):
    data_loss = trainrun.train_LM_GD(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
    }

    summary[key] = results


def train_all(optimizer, trainloader, net_init_parameters, classes):
    summary = {}
    timer = timing.Timing()

    for key, opt in optimizer.items():
        with timer(key):
            if deepsplitting.utils.misc.is_splitting(opt):
                train_splitting(opt, key, trainloader, net_init_parameters, summary)
            else:
                train_lm(opt, key, trainloader, net_init_parameters, summary)

        if global_config.cfg.loss_type == 'ls':
            testrun.test_ls(opt.net, trainloader, classes)
        elif global_config.cfg.loss_type == 'nll':
            testrun.test_nll(opt.net, trainloader)

    deepsplitting.utils.evaluate.plot_summary(summary, timer, optimizer, filename='results',
                                              folder=global_config.cfg.results_folder)

    deepsplitting.utils.evaluate.save_summary(optimizer, summary, timer)


if __name__ == '__main__':
    main()
