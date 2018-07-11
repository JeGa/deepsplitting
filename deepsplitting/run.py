import logging

import deepsplitting.optimizer.gradient_descent as GD
import deepsplitting.optimizer.gradient_descent_armijo as GDA
import deepsplitting.optimizer.llc as LLC
import deepsplitting.optimizer.prox_descent as ProxDescent
import deepsplitting.optimizer.levenberg_marquardt as LM

import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun
import deepsplitting.utils.timing as timing

from deepsplitting.utils.misc import *
from deepsplitting.optimizer.base import Hyperparams

optimizer_params_ls = {
    'LLC': Hyperparams(M=0.001, factor=10, rho=1),
    'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    'LM': Hyperparams(M=0.001, factor=10),
    'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    'GD': Hyperparams(lr=0.0005)
}

optimizer_params_nll = {
    'LLC': Hyperparams(M=0.001, factor=10, rho=1),
    'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    'GD': Hyperparams(lr=0.005)
}


def main():
    logging.basicConfig(level=logging.INFO)

    params = {
        'loss_type': 'nll',  # 'ls' or 'nll'
        'activation_type': 'relu',  # 'relu' or 'sigmoid'
        'resutls_folder': '../results'
    }

    net, trainloader, training_batch_size = initializer.ff_spirals(params['loss_type'], params['activation_type'])

    optimizer_params = optimizer_params_nll

    optimizer = {
        'LLC': LLC.Optimizer(net, N=training_batch_size, hyperparams=optimizer_params['LLC']),
        'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),  # Not with nll.
        'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        'GD': GD.Optimizer(net, hyperparams=optimizer_params['GD'])}

    if params['loss_type'] == 'ls':
        optimizer['LM'] = LM.Optimizer(net, hyperparams=optimizer_params['LM'])

    losses = trainrun.train(trainloader, optimizer['ProxDescent'], 10)
    plot_loss_curve(losses)
    testrun.test_nll(net, trainloader)

    # train_all(optimizer, trainloader, params)


def train_all(optimizer, trainloader, params):
    summary = {}
    timer = deepsplitting.utils.timing.Timing()

    for key, opt in optimizer.items():
        timer.start()
        losses = trainrun.train(trainloader, opt, 30)
        timer.stop(key)

        if params['loss_type'] == 'ls':
            testrun.test_ls(opt.net, trainloader, 2)
        elif params['loss_type'] == 'nll':
            testrun.test_nll(opt.net, trainloader)

        # plot_loss_curve(losses, key)

        summary[key] = losses

    plot_summary(summary, timer, title='Loss: ' + params['loss_type'] + ', activation: ' + params['activation_type'])


if __name__ == '__main__':
    main()
