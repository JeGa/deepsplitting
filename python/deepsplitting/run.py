import matplotlib

matplotlib.use('Agg')

import logging
import progressbar

progressbar.streams.wrap_stderr()

import deepsplitting.utils.evaluate
import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.timing as timing
import deepsplitting.utils.misc
import deepsplitting.config as config_file
import deepsplitting.utils.global_config as global_config


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

    optimizer = {params.key: params.create(net, N=training_batch_size, hyperparams=params.params) for params in
                 optimizer_params if params.on}

    # To have the same network parameter initialization for all nets.
    net_init_parameters = next(iter(optimizer.values())).save_params()

    train_all(optimizer, trainloader, net_init_parameters, classes)


def train_splitting(opt, key, trainloader, net_init_parameters, summary):
    data_loss, lagrangian, correct = trainrun.train_splitting(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss,
        'lagrangian': lagrangian
    }

    if len(correct) != 0:
        results['correct'] = correct

    summary[key] = results


def train_lm(opt, key, trainloader, net_init_parameters, summary):
    data_loss, correct = trainrun.train_LM_GD(trainloader, opt, net_init_parameters)

    results = {
        'data_loss': data_loss
    }

    if len(correct) != 0:
        results['correct'] = correct

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
            correct, samples = testrun.test_ls(opt.net, trainloader, classes)
        elif global_config.cfg.loss_type == 'nll':
            correct, samples = testrun.test_nll(opt.net, trainloader)
        else:
            raise ValueError()

        print("{} of {} correctly classified.".format(correct, samples))

    deepsplitting.utils.evaluate.save_summary(optimizer, summary, timer)


if __name__ == '__main__':
    main()
