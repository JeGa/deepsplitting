from collections import namedtuple

from deepsplitting.optimizer.base import Hyperparams

import deepsplitting.optimizer.splitting.batched_levenberg_marquardt as sbLM
import deepsplitting.optimizer.lm.batched_levenberg_marquardt as bLM
import deepsplitting.optimizer.splitting.batched_gradient_descent as sbGD
import deepsplitting.optimizer.gd.gradient_descent as GD
import deepsplitting.optimizer.gd.gradient_descent_armijo as GDA
import deepsplitting.optimizer.other.prox_descent as ProxDescent
import deepsplitting.optimizer.other.proxprop as ProxProp

ParamsEntry = namedtuple('ParamsEntry', ['on', 'key', 'create', 'params'])

optimizer_params_autoencoder = [
    ParamsEntry(
        True, 'bGD_fix', GD.Optimizer_batched,
        Hyperparams(lr=1e-5, vanstep=False)),
    ParamsEntry(
        False, 'GDA', GDA.Optimizer,
        Hyperparams(beta=0.5, gamma=1e-6)),
]

# sbGD parameter search. ======================================================

optimizer_params_ls_paramsearch_sbGD5 = [
    ParamsEntry(
        True, '5_sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        True, '5_sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=5e-4, vanstep=False)),
    ParamsEntry(
        True, '5_sbGD_fix_1e-4', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=1e-4, vanstep=False)),
    ParamsEntry(
        True, '5_sbGD_fix_5e-5', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=5e-5, vanstep=False)),
]

optimizer_params_ls_paramsearch_sbGD10_p2mb = [
    ParamsEntry(
        True, 'p2mb_10_sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=100)),
    ParamsEntry(
        True, 'p2mb_10_sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-4, vanstep=False, primal2_batches=100)),
    ParamsEntry(
        True, 'p2mb_10_sbGD_fix_1e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-4, vanstep=False, primal2_batches=100)),
    ParamsEntry(
        True, 'p2mb_10_sbGD_fix_5e-5', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-5, vanstep=False, primal2_batches=100)),
]

optimizer_params_ls_paramsearch_sbGD30 = [
    ParamsEntry(
        True, '30_sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        True, '30_sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-4, vanstep=False)),
    ParamsEntry(
        True, '30_sbGD_fix_1e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-4, vanstep=False)),
    ParamsEntry(
        True, '30_sbGD_fix_5e-5', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-5, vanstep=False)),
]

optimizer_params_ls_paramsearch_sbGD50 = [
    ParamsEntry(
        True, '50_sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        True, '50_sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-4, vanstep=False)),
    ParamsEntry(
        True, '50_sbGD_fix_1e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-4, vanstep=False)),
    ParamsEntry(
        True, '50_sbGD_fix_5e-5', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-5, vanstep=False)),
]

optimizer_params_ls_paramsearch_sbGD100 = [
    ParamsEntry(
        True, '100_sbGD_fix_1e-3', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        True, '100_sbGD_fix_5e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-4, vanstep=False)),
    ParamsEntry(
        True, '100_sbGD_fix_1e-4', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-4, vanstep=False)),
    ParamsEntry(
        True, '100_sbGD_fix_5e-5', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=5e-5, vanstep=False)),
]

# bGD parameter search ========================================================

optimizer_params_ls_paramsearch_bGD = [
    ParamsEntry(
        True, '10_bGD_fix_1e-3', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=False)),
    ParamsEntry(
        True, '10_bGD_fix_5e-4', GD.Optimizer_batched,
        Hyperparams(lr=5e-4, vanstep=False)),
    ParamsEntry(
        True, '10_bGD_fix_1e-4', GD.Optimizer_batched,
        Hyperparams(lr=1e-4, vanstep=False)),
    ParamsEntry(
        True, '10_bGD_fix_5e-5', GD.Optimizer_batched,
        Hyperparams(lr=5e-5, vanstep=False)),
]

# For report. =================================================================

# Spirals dataset and nll.
report_optimizer_params_nll = [
    ParamsEntry(
        True, 'spirals_bGD_nll', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=False)),

    ParamsEntry(
        False, 'spirals_sbLM_damping_nll', sbLM.Optimizer_damping,
        Hyperparams(rho=1, rho_add=0, subsample_factor=1, cg_iter=10, M=0.001, factor=10, primal2_batches=1)),
    ParamsEntry(
        False, 'spirals_sbLM_armijo_nll', sbLM.Optimizer_armijo,
        Hyperparams(rho=1, rho_add=0, subsample_factor=1, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4,
                    primal2_batches=1)),

    ParamsEntry(
        False, 'spirals_sbGD_nll_rho1', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=5e-4, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        False, 'spirals_sbGD_nll_rho5', sbGD.Optimizer,
        Hyperparams(rho=5, rho_add=0, stepsize=5e-4, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        False, 'spirals_sbGD_nll_rho10', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        False, 'spirals_sbGD_nll_rho50', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        True, 'spirals_sbGD_nll_rho100', sbGD.Optimizer,
        Hyperparams(rho=100, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1))
]

# MNIST and LS loss.
report_optimizer_params_ls = [
    ParamsEntry(
        True, 'mnist_bGD_fix', GD.Optimizer_batched,
        Hyperparams(lr=1e-3, vanstep=False)),

    ParamsEntry(
        True, 'mnist_sbGD_nll_rho5', sbGD.Optimizer,
        Hyperparams(rho=5, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        True, 'mnist__sbGD_nll_rho10', sbGD.Optimizer,
        Hyperparams(rho=10, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1)),
    ParamsEntry(
        True, 'mnist__sbGD_nll_rho50', sbGD.Optimizer,
        Hyperparams(rho=50, rho_add=0, stepsize=1e-3, vanstep=False, primal2_batches=1)),

    ParamsEntry(
        False, 'mnist_bLM_vanstep', bLM.Optimizer_vanstep,
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, stepsize=1e-3, stepsize_fix=True)),
    ParamsEntry(
        False, 'mnist_bLM_armijo', bLM.Optimizer_armijo,
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=1e-3)),
]

# All optimizer ===============================================================

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
        False, 'sbGD_fix', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-3, vanstep=False)),
    ParamsEntry(
        False, 'sbGD_vanstep', sbGD.Optimizer,
        Hyperparams(rho=1, rho_add=0, stepsize=1e-3, min_stepsize=1e-6, vanstep=True)),

    # Batched Levenberg-Marquardt (only works with LS loss).
    ParamsEntry(
        False, 'bLM_damping', bLM.Optimizer_damping,
        Hyperparams(subsample_factor=1, cg_iter=8, M=0.001, factor=5)),
    ParamsEntry(
        True, '5_bLM_armijo', bLM.Optimizer_armijo,
        Hyperparams(subsample_factor=1, cg_iter=10, delta=1, eta=0.5, beta=0.5, gamma=10e-4)),
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
