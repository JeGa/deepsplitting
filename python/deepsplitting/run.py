import matplotlib

matplotlib.use('Agg')

import logging

import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.timing as timing
import deepsplitting.utils.misc
import deepsplitting.config as cfg
import deepsplitting.utils.global_config as global_config

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM

from deepsplitting.utils.misc import *

global_config.cfg = cfg.local_cfg


def main():
    logging.basicConfig(level=logging.INFO)

    deepsplitting.utils.misc.make_results_folder(global_config.cfg)

    net, trainloader, training_batch_size, classes = initializer.cnn_mnist(cfg.params.loss_type,
                                                                           cfg.params.activation_type)

    if cfg.params.loss_type == 'ls':
        optimizer_params = cfg.optimizer_params_ls
    elif cfg.params.loss_type == 'nll':
        optimizer_params = cfg.optimizer_params_nll
    else:
        raise ValueError("Unsupported loss type.")

    optimizer = {
        'sbLM_damping': sbLM.Optimizer_damping(net, N=training_batch_size, hyperparams=optimizer_params['sbLM_damping']),

        # 'LLC_fix': LLC.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['LLC_fix']),
        # 'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),
        # 'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        # 'GD': GD.Optimizer(net, hyperparams=optimizer_params['GD']),
        # 'ProxProp': ProxProp.Optimizer(net, hyperparams=optimizer_params['ProxProp'])
    }

    # if params.loss_type == 'ls':
    #    optimizer['LM'] = LM.Optimizer(net, hyperparams=optimizer_params['LM'])

    # To have the same network parameter initialization for all nets.
    net_init_parameters = next(iter(optimizer.values())).save_params()

    train_all(optimizer, trainloader, cfg.params, net_init_parameters, optimizer_params, classes)


def train(opt, key, trainloader, net_init_parameters, summary):
    # if deepsplitting.utils.misc.is_llc(opt):
    data_loss, lagrangian = trainrun.train_llc(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
        'lagrangian': lagrangian
    }

    summary[key] = results


# else:
#    raise NotImplementedError()


def train_all(optimizer, trainloader, params, net_init_parameters, optimizer_params, classes):
    summary = {}
    timer = timing.Timing()

    for key, opt in optimizer.items():
        with timer(key):
            train(opt, key, trainloader, net_init_parameters, summary)

        if params.loss_type == 'ls':
            testrun.test_ls(opt.net, trainloader, classes)
        elif params.loss_type == 'nll':
            testrun.test_nll(opt.net, trainloader)

    # TODO plot_summary(summary, timer, name='results',
    #             title='Loss: ' + params.loss_type + ', activation: ' + params.activation_type)

    # TODO save_summary(summary, timer, optimizer_params)


if __name__ == '__main__':
    main()
