import torch
import yaml
import logging

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.base import Hyperparams

local_cfg = global_config.GlobalParams(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
    device=torch.device('cpu'),

    epochs=1,
    training_batch_size=10,
    training_samples=50,  # Take subset of training set.
    forward_chunk_size_factor=1,

    datatype=torch.double,
    seed=123,

    logging=-1,  # To enable: logging.INFO
    results_folder='results',
    results_subfolders={'data': 'data'}
)

server_cfg = global_config.GlobalParams(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
    device=torch.device('cuda'),

    epochs=20,
    training_batch_size=1000,
    training_samples=1000,  # Take subset of training set.
    forward_chunk_size_factor=0.1,

    datatype=torch.double,
    seed=123,

    logging=logging.INFO,  # To enable: logging.INFO, to disable set to -1.
    results_folder='results',
    results_subfolders={'data': 'data'}
)

# config_file.server_cfg or config_file.local_cfg.
global_config.cfg = local_cfg


# Required if config has objects that can not be serialized using yaml.
def torch_double_representer(dumper, data):
    return dumper.represent_scalar('!torch.float64', str(data))


yaml.add_representer(torch.float64, torch_double_representer)


def torch_device_representer(dumper, data):
    return dumper.represent_scalar('!torch.device', str(data))


yaml.add_representer(torch.device, torch_device_representer)


# Required if config has objects that can not be serialized using yaml.
def torch_double_representer(dumper, data):
    return dumper.represent_scalar('!torch.double', str(data))


yaml.add_representer(torch.dtype, torch_double_representer)

if not isinstance(global_config.cfg, global_config.GlobalParams):
    raise ValueError("Global config wrong instance.")

optimizer_params_ls = {
    # Splitting with different batched LM steps.
    'sbLM_damping':
        Hyperparams(rho=10, rho_add=0, subsample_factor=1, cg_iter=15, M=0.001, factor=10),
    'sbLM_armijo':
        Hyperparams(rho=1, rho_add=0, subsample_factor=0.5, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4),
    'sbLM_vanstep':
        Hyperparams(rho=1, rho_add=0, subsample_factor=1, cg_iter=15, delta=1, eta=0.5, stepsize=1e-3),

    # Splitting with batched GD step.
    'sbGD_fix':
        Hyperparams(rho=10, rho_add=0, subsample_factor=0.7, stepsize=1e-3, vanstep=False),
    'sbGD_vanstep':
        Hyperparams(rho=10, rho_add=0, subsample_factor=0.7, stepsize=1e-3, vanstep=True),

    # Batched Levenberg-Marquardt (only works with LS loss).
    'bLM_damping':
        Hyperparams(subsample_factor=1, cg_iter=20, M=0.001, factor=5),
    'bLM_armijo':
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4),
    'bLM_vanstep':
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5),

    # Stochastic (batched) gradient descent.
    'bGD_fix':
        Hyperparams(lr=1e-3, vanstep=False),
    'bGD_vanstep':
        Hyperparams(lr=1e-3, vanstep=True),

    # Other stuff not used currently.
    'GDA':
        Hyperparams(beta=0.5, gamma=10 ** -4),
    'ProxDescent':
        Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    'ProxProp':
        Hyperparams(tau=0.005, tau_theta=5)
}

optimizer_params_nll = {
    # 'LLC': Hyperparams(M=0.001, factor=10, rho=35, rho_add=1),
    # 'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    # 'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    # 'GD': Hyperparams(lr=0.005),
    # 'ProxProp': Hyperparams(tau=0.005, tau_theta=10)
}
