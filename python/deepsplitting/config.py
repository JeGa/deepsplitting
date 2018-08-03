import torch

import deepsplitting.utils.global_config as global_config

from deepsplitting.optimizer.base import Hyperparams
from deepsplitting.utils.misc import Params

params = Params(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
)

local_cfg = global_config.GlobalParams(
    device=torch.device('cpu'),
    training_batch_size=10,
    epochs=100,
    training_samples=10,  # Take subset of training set.
    forward_chunk_size_factor=1,
    results_folder='results',
    results_subfolders={'data': 'data'}
)

server_cfg = global_config.GlobalParams(
    device=torch.device('cuda'),
    training_batch_size=100,
    epochs=1,
    training_samples=-1,  # Take subset of training set.
    forward_chunk_size_factor=0.01,
    results_folder='results',
    results_subfolders={'data': 'data'}
)

# config_file.server_cfg or config_file.local_cfg.
global_config.cfg = server_cfg

if not isinstance(global_config.cfg, Params):
    raise ValueError("Global config wrong instance.")

optimizer_params_ls = {
    'sbLM_damping': Hyperparams(M=0.001, factor=10, rho=5, rho_add=0, subsample_factor=0.7, cg_iter=10),

    # 'LLC_fix': Hyperparams(M=0.001, factor=10, rho=5, rho_add=0),
    # 'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    # 'LM': Hyperparams(M=0.001, factor=10),
    # 'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    # 'GD': Hyperparams(lr=0.0005),
    # 'ProxProp': Hyperparams(tau=0.005, tau_theta=5)
}

optimizer_params_nll = {
    # 'LLC': Hyperparams(M=0.001, factor=10, rho=35, rho_add=1),
    # 'ProxDescent': Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3),
    # 'GDA': Hyperparams(beta=0.5, gamma=10 ** -4),
    # 'GD': Hyperparams(lr=0.005),
    # 'ProxProp': Hyperparams(tau=0.005, tau_theta=10)
}
