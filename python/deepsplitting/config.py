from deepsplitting.config_params import *
from deepsplitting.config_global_params import *

# server_cfg or local_cfg.
global_config.cfg = server_cfg

# Change here which optimizers to use.
ls_params = optimizer_params_ls_paramsearch_sbGD10_p2mb
nll_params = None

# Just for checking.
if not isinstance(global_config.cfg, global_config.GlobalParams):
    raise ValueError("Global config wrong instance.")
