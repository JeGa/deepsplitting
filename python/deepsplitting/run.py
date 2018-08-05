import matplotlib

matplotlib.use('Agg')

import logging

import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.timing as timing
import deepsplitting.utils.misc
import deepsplitting.config as config_file
import deepsplitting.utils.global_config as global_config

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM
import deepsplitting.optimizer.splitting.batched_gradient_descent as sbGD
import deepsplitting.optimizer.lm.batched_levenberg_marquardt as bLM
import deepsplitting.optimizer.gd.gradient_descent as GD


def main():
    logging.basicConfig(level=logging.INFO)

    deepsplitting.utils.misc.make_results_folder(global_config.cfg)

    net, trainloader, training_batch_size, classes = initializer.cnn_mnist(global_config.cfg.loss_type,
                                                                           global_config.cfg.activation_type)

    if global_config.cfg.loss_type == 'ls':
        optimizer_params = config_file.optimizer_params_ls
    elif global_config.cfg.loss_type == 'nll':
        optimizer_params = config_file.optimizer_params_nll
    else:
        raise ValueError("Unsupported loss type.")

    optimizer = {
        # 'sbLM_damping': sbLM.Optimizer_damping(net, N=training_batch_size,
        #                                       hyperparams=optimizer_params['sbLM_damping']),

        # 'sbLM_armijo': sbLM.Optimizer_armijo(net, N=training_batch_size,
        #                                     hyperparams=optimizer_params['sbLM_armijo']),

        # 'sbLM_vanstep': sbLM.Optimizer_vanstep(net, N=training_batch_size,
        #                                      hyperparams=optimizer_params['sbLM_vanstep']),

        # 'bLM_damping': bLM.Optimizer_damping(net, N=training_batch_size,
        #                                     hyperparams=optimizer_params['bLM_damping']),

        'bLM_armijo': bLM.Optimizer_armijo(net, N=training_batch_size,
                                           hyperparams=optimizer_params['bLM_armijo']),

        # 'sbGD': sbGD.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['sbGD']),

        # 'LLC_fix': LLC.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['LLC_fix']),
        # 'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),
        # 'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        # 'GD': GD.Optimizer(net, hyperparams=optimizer_params['GD']),
        # 'ProxProp': ProxProp.Optimizer(net, hyperparams=optimizer_params['ProxProp'])
    }

    # trainrun.train(trainloader, )

    # if params.loss_type == 'ls':
    #    optimizer['LM'] = LM.Optimizer(net, hyperparams=optimizer_params['LM'])

    # To have the same network parameter initialization for all nets.
    net_init_parameters = next(iter(optimizer.values())).save_params()

    train_all(optimizer, trainloader, net_init_parameters, classes)


def train_splitting(opt, key, trainloader, net_init_parameters, summary):
    # TODO: if deepsplitting.utils.misc.is_llc(opt):
    data_loss, lagrangian = trainrun.train_splitting(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
        'lagrangian': lagrangian
    }

    summary[key] = results


def train_lm(opt, key, trainloader, net_init_parameters, summary):
    data_loss = trainrun.train_LM(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
    }

    summary[key] = results


def train_all(optimizer, trainloader, net_init_parameters, classes):
    summary = {}
    timer = timing.Timing()

    for key, opt in optimizer.items():
        with timer(key):
            # train_splitting(opt, key, trainloader, net_init_parameters, summary)
            train_lm(opt, key, trainloader, net_init_parameters, summary)

        if global_config.cfg.loss_type == 'ls':
            testrun.test_ls(opt.net, trainloader, classes)
        elif global_config.cfg.loss_type == 'nll':
            testrun.test_nll(opt.net, trainloader)

    deepsplitting.utils.misc.plot_summary(summary, timer, optimizer, filename='results',
                                          folder=global_config.cfg.results_folder)

    # TODO save_summary(summary, timer, optimizer_params)


if __name__ == '__main__':
    main()
