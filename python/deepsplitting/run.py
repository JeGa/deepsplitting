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
import deepsplitting.optimizer.gd.gradient_descent as GD

def main():
    logging.basicConfig(level=logging.INFO)

    deepsplitting.utils.misc.make_results_folder(global_config.cfg)

    net, trainloader, training_batch_size, classes = initializer.cnn_mnist(config_file.params.loss_type,
                                                                           config_file.params.activation_type)

    if config_file.params.loss_type == 'ls':
        optimizer_params = config_file.optimizer_params_ls
    elif config_file.params.loss_type == 'nll':
        optimizer_params = config_file.optimizer_params_nll
    else:
        raise ValueError("Unsupported loss type.")

    optimizer = {
        'sbLM_damping': sbLM.Optimizer_damping(net, N=training_batch_size,
                                               hyperparams=optimizer_params['sbLM_damping']),

        # 'LLC_fix': LLC.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['LLC_fix']),
        # 'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),
        # 'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        #'GD': GD.Optimizer(net, hyperparams=optimizer_params['GD']),
        # 'ProxProp': ProxProp.Optimizer(net, hyperparams=optimizer_params['ProxProp'])
    }

    #trainrun.train(trainloader, )

    # if params.loss_type == 'ls':
    #    optimizer['LM'] = LM.Optimizer(net, hyperparams=optimizer_params['LM'])

    # To have the same network parameter initialization for all nets.
    net_init_parameters = next(iter(optimizer.values())).save_params()

    train_all(optimizer, trainloader, config_file.params, net_init_parameters, classes)


def train(opt, key, trainloader, net_init_parameters, summary):
    # TODO: if deepsplitting.utils.misc.is_llc(opt):
    data_loss, lagrangian = trainrun.train_llc(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
        'lagrangian': lagrangian
    }

    summary[key] = results


def train_all(optimizer, trainloader, params, net_init_parameters, classes):
    summary = {}
    timer = timing.Timing()

    for key, opt in optimizer.items():
        with timer(key):
            train(opt, key, trainloader, net_init_parameters, summary)

        if params.loss_type == 'ls':
            testrun.test_ls(opt.net, trainloader, classes)
        elif params.loss_type == 'nll':
            testrun.test_nll(opt.net, trainloader)

    deepsplitting.utils.misc.plot_summary(summary, timer, optimizer, params,
                                          filename='results', folder=global_config.cfg.results_folder)

    # TODO save_summary(summary, timer, optimizer_params)


if __name__ == '__main__':
    main()
