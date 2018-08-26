from collections import namedtuple

from deepsplitting.optimizer.base import Hyperparams

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM
import deepsplitting.optimizer.lm.batched_levenberg_marquardt as bLM
import deepsplitting.optimizer.splitting.batched_gradient_descent as sbGD
import deepsplitting.optimizer.gd.gradient_descent as GD
import deepsplitting.optimizer.other.prox_descent as ProxDescent
import deepsplitting.optimizer.other.proxprop as ProxProp

ParamsEntry = namedtuple('ParamsEntry', ['on', 'key', 'create', 'params'])

optimizer_params_ls_paramsearch_sbGD = [
    ParamsEntry(
        True, 'sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-3, vanstep=False)),

    ParamsEntry(
        True, 'sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=5e-4, vanstep=False)),

    ParamsEntry(
        True, 'sbGD_fix_7e-4', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-4, vanstep=False)),

    ParamsEntry(
        True, 'sbGD_fix_9e-4', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-5, vanstep=False))
]

optimizer_params_ls_paramsearch_bGD = [
    ParamsEntry(
        True, 'bGD_fix_1e-3', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=False)),
    ParamsEntry(
        True, 'bGD_fix_1e-4', GD.Optimizer_batched,
        Hyperparams(lr=1e-4, vanstep=False)),
    ParamsEntry(
        True, 'bGD_fix_1e-5', GD.Optimizer_batched,
        Hyperparams(lr=1e-5, vanstep=False)),
    ParamsEntry(
        True, 'bGD_fix_1e-6', GD.Optimizer_batched,
        Hyperparams(lr=1e-6, vanstep=False)),
    ParamsEntry(
        True, 'bGD_fix_1e-7', GD.Optimizer_batched,
        Hyperparams(lr=1e-7, vanstep=False))
]

optimizer_params_ls = [
    # Splitting with different batched LM steps.
    ParamsEntry(
        False, 'sbLM_damping', sbLM.Optimizer_damping,
        Hyperparams(rho=10, rho_add=0, subsample_factor=1, cg_iter=8, M=0.001, factor=10)),
    ParamsEntry(
        False, 'sbLM_armijo', sbLM.Optimizer_armijo,
        Hyperparams(rho=1, rho_add=0, subsample_factor=0.5, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4)),
    ParamsEntry(
        False, 'sbLM_vanstep', sbLM.Optimizer_vanstep,
        Hyperparams(rho=1, rho_add=0, subsample_factor=1, cg_iter=15, delta=1, eta=0.5, stepsize=1e-3,
                    stepsize_fix=True)),

    # Splitting with batched GD step.
    ParamsEntry(
        True, 'sbGD_fix', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        False, 'sbGD_vanstep', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-3, min_stepsize=1e-6, vanstep=True)),

    # Batched Levenberg-Marquardt (only works with LS loss).
    ParamsEntry(
        False, 'bLM_damping', bLM.Optimizer_damping,
        Hyperparams(subsample_factor=1, cg_iter=8, M=0.001, factor=5)),
    ParamsEntry(
        False, 'bLM_armijo', bLM.Optimizer_armijo,
        Hyperparams(subsample_factor=0.5, cg_iter=20, delta=1, eta=0.5, beta=0.5, gamma=10e-4)),
    ParamsEntry(
        False, 'bLM_vanstep', bLM.Optimizer_vanstep,
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, stepsize=1e-4, stepsize_fix=True)),

    # Stochastic (batched) gradient descent.
    ParamsEntry(
        False, 'bGD_fix', GD.Optimizer_batched,
        Hyperparams(lr=1e-5, vanstep=False)),
    ParamsEntry(
        False, 'bGD_vanstep', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=True)),
]

# Other stuff not used currently.
optimizer_params_misc = [
    ParamsEntry(
        False, 'ProxDescent', ProxDescent.Optimizer,
        Hyperparams(tau=1.5, sigma=0.5, mu_min=0.3)),
    ParamsEntry(
        False, 'ProxProp', ProxProp.Optimizer,
        Hyperparams(tau=0.005, tau_theta=5))
]
