import logging

import deepsplitting.optimizer.gradient_descent as GD
import deepsplitting.optimizer.gradient_descent_armijo as GDA
import deepsplitting.optimizer.llc as LLC
import deepsplitting.optimizer.prox_descent as ProxDescent
import deepsplitting.optimizer.levenberg_marquardt as LM

import deepsplitting.utils.initializer as initializer
import deepsplitting.utils.trainrun as trainrun
import deepsplitting.utils.testrun as testrun

from deepsplitting.utils.misc import *
from deepsplitting.optimizer.base import Hyperparams


def main():
    logging.basicConfig(level=logging.INFO)

    params = {
        'loss_type': 'ls',
        'activation_type': 'relu',
        'resutls_folder': '../results'
    }

    net, trainloader, training_batch_size = initializer.ff_spirals(params['loss_type'], params['activation_type'])

    optimizer_params = {
        'LLC': Hyperparams(M=0.001, factor=10, rho=1),
        'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
        'LM': Hyperparams(M=0.001, factor=10),
        'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
        'GD': Hyperparams(lr=0.0005)
    }

    optimizer = {
        'LLC': LLC.Optimizer(net, N=training_batch_size,
                             debug=True, hyperparams=optimizer_params['LLC']),  # Not with nll.
        'ProxDescent': ProxDescent.Optimizer(net, hyperparams=optimizer_params['ProxDescent']),  # not with nll.
        'LM': LM.Optimizer(net, hyperparams=optimizer_params['LM']),  # Only works with ls loss.
        'GDA': GDA.Optimizer(net, hyperparams=optimizer_params['GDA']),
        'GD': GD.Optimizer(net, hyperparams=optimizer_params['GD'])}

    # losses = train(trainloader, optimizer['ProxDescent'], 30)
    # plot_loss_curve(losses)

    train_all(optimizer, trainloader, params['loss_type'])


def train_all(optimizer, trainloader, loss_type):
    summary = {}

    for key, opt in optimizer.items():
        params = opt.save_params()

        losses = trainrun.train(trainloader, opt, 30)

        if loss_type == 'ls':
            testrun.test_ls(opt.net, trainloader, 2)

        # plot_loss_curve(losses, key)

        summary[key] = losses

        opt.restore_params(params)

    plot_summary(summary, title='Loss: ' + loss_type)


if __name__ == '__main__':
    main()
