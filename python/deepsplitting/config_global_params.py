import torch
import yaml

import deepsplitting.utils.global_config as global_config

local_cfg = global_config.GlobalParams(
    loss_type='ls',  # 'ls' or 'nll'.
    activation_type='relu',  # 'relu' or 'sigmoid'.
    device=torch.device('cpu'),
    classes=10,

    epochs=10,
    training_batch_size=10,
    training_samples=50,  # Take subset of training set.
    forward_chunk_size_factor=1,
    test_interval=-1,  # -1 to disable.

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

    epochs=1,
    training_batch_size=50,
    training_samples=-1,  # Take subset of training set.
    forward_chunk_size_factor=0.1,
    test_interval=-1,  # -1 to disable.

    datatype=torch.double,
    seed=123,

    logging=-1,  # To enable: logging.INFO, to disable set to -1.
    results_folder='results',
    results_subfolders={'data': 'data', 'plots': 'plots'}
)


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


yaml_custom_types()
