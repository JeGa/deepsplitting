from deepsplitting.config_params import *
from deepsplitting.config_global_params import *

# server_cfg or local_cfg.
global_config.cfg = server_cfg

# Optimizer parameters.
params = report_optimizer_params_ls

# Just for checking.
if not isinstance(global_config.cfg, global_config.GlobalParams):
    raise ValueError("Global config wrong instance.")
