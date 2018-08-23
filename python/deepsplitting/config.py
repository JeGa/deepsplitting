import torch
import yaml
from collections import namedtuple

import deepsplitting.utils.global_config as global_config
from deepsplitting.optimizer.base import Hyperparams

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM
import deepsplitting.optimizer.lm.batched_levenberg_marquardt as bLM
import deepsplitting.optimizer.splitting.batched_gradient_descent as sbGD
import deepsplitting.optimizer.gd.gradient_descent as GD

local_cfg = global_config.GlobalParams(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
    device=torch.device('cpu'),
    classes=10,

    epochs=5,
    training_batch_size=10,
    training_samples=50,  # Take subset of training set.
    forward_chunk_size_factor=1,
    test_interval=5,  # -1 to disable.

    datatype=torch.double,
    seed=123,

    logging=-1,  # To enable: logging.INFO
    results_folder='results',
    results_subfolders={'data': 'data', 'plots': 'plots'}
)

server_cfg = global_config.GlobalParams(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
    device=torch.device('cuda'),
    classes=10,

    epochs=20,
    training_batch_size=50,
    training_samples=1000,  # Take subset of training set.
    forward_chunk_size_factor=0.1,
    test_interval=25,  # -1 to disable.

    datatype=torch.double,
    seed=123,

    logging=-1,  # To enable: logging.INFO, to disable set to -1.
    results_folder='results',
    results_subfolders={'data': 'data', 'plots': 'plots'}
)

# config_file.server_cfg or config_file.local_cfg.
global_config.cfg = server_cfg


def yaml_custom_types():
    """
    Required if config has objects that can not be serialized using yaml.
    """

    def torch_double_representer(dumper, data):
        return dumper.represent_scalar('!torch.float64', str(data))

    yaml.add_representer(torch.float64, torch_double_representer)

    def torch_device_representer(dumper, data):
        return dumper.represent_scalar('!torch.device', str(data))

    yaml.add_representer(torch.device, torch_device_representer)

    def torch_double_representer(dumper, data):
        return dumper.represent_scalar('!torch.double', str(data))

    yaml.add_representer(torch.dtype, torch_double_representer)

    def str_constructor(loader, node):
        return node.value

    yaml.add_constructor('!torch.float64', str_constructor)
    yaml.add_constructor('!torch.device', str_constructor)
    yaml.add_constructor('!torch.double', str_constructor)


if not isinstance(global_config.cfg, global_config.GlobalParams):
    raise ValueError("Global config wrong instance.")

yaml_custom_types()

ParamsEntry = namedtuple('ParamsEntry', ['on', 'key', 'create', 'params'])

optimizer_params_ls = [
    # Splitting with different batched LM steps.
    ParamsEntry(
        False, 'sbLM_damping', sbLM.Optimizer_damping,
        Hyperparams(rho=10, rho_add=0, subsample_factor=1, cg_iter=8, M=0.001, factor=10)),
    ParamsEntry(
        True, 'sbLM_armijo', sbLM.Optimizer_armijo,
        Hyperparams(rho=1, rho_add=0, subsample_factor=0.5, cg_iter=20, delta=1, eta=0.5, beta=0.5, gamma=10e-4)),
    ParamsEntry(
        False, 'sbLM_vanstep', sbLM.Optimizer_vanstep,
        Hyperparams(rho=1, rho_add=0, subsample_factor=1, cg_iter=15, delta=1, eta=0.5, stepsize=1e-3,
                    stepsize_fix=True)),

    # Splitting with batched GD step.
    ParamsEntry(
        False, 'sbGD_fix', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, subsample_factor=0.5, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        False, 'sbGD_vanstep', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, subsample_factor=0.5, stepsize=1e-3, min_stepsize=1e-6, vanstep=True)),

    # Batched Levenberg-Marquardt (only works with LS loss).
    ParamsEntry(
        False, 'bLM_damping', bLM.Optimizer_damping,
        Hyperparams(subsample_factor=1, cg_iter=8, M=0.001, factor=5)),
    ParamsEntry(
        False, 'bLM_armijo', bLM.Optimizer_armijo,
        Hyperparams(subsample_factor=0.5, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4)),
    ParamsEntry(
        False, 'bLM_vanstep', bLM.Optimizer_vanstep,
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, stepsize=1e-4, stepsize_fix=True)),

    # Stochastic (batched) gradient descent.
    ParamsEntry(
        False, 'bGD_fix', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=False)),
    ParamsEntry(
        False, 'bGD_vanstep', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=True)),

    # Other stuff not used currently.
    # ParamsEntry(
    #    False, 'GDA',
    #    Hyperparams(beta=0.5, gamma=10 ** -4)),
    # ParamsEntry(
    #    False, 'ProxDescent',
    #    Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3)),
    # ParamsEntry(
    #    False, 'ProxProp',
    #    Hyperparams(tau=0.005, tau_theta=5))
]

optimizer_params_nll = {
    # 'LLC': Hyperparams(M=0.001, factor=10, rho=35, rho_add=1),
    # 'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    # 'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    # 'GD': Hyperparams(lr=0.005),
    # 'ProxProp': Hyperparams(tau=0.005, tau_theta=10)
}
