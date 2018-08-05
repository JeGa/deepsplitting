import torch
import numpy as np
from enum import Enum, auto

import deepsplitting.utils.global_config as global_config
from deepsplitting.utils.global_config import Params


class Hyperparams(Params):
    def __init__(self, **params):
        super(Hyperparams, self).__init__(**params)


class Initializer(Enum):
    RANDN = auto()
    DEBUG = auto()
    FROM_PARAMS = auto()


class BaseOptimizer:
    def __init__(self, net, hyperparams):
        self.net = net
        self.hyperparams = hyperparams

        self.init_parameters(Initializer.RANDN)

    def init_parameters(self, initializer, parameters=None):
        if initializer is Initializer.FROM_PARAMS and parameters is None:
            raise ValueError("No parameters list given.")

        if initializer is Initializer.DEBUG:
            def init_fun(submodule):
                if type(submodule) == torch.nn.Linear:
                    submodule.weight.data.fill_(1)
                    submodule.bias.data.fill_(1)

            self.net.apply(init_fun)
        elif initializer is Initializer.RANDN:
            def init_fun(submodule):
                if type(submodule) == torch.nn.Linear:
                    torch.nn.init.normal_(submodule.weight)
                    torch.nn.init.normal_(submodule.bias)

                    submodule.weight.data.mul_(0.1)
                    submodule.bias.data.mul_(0.1)

            self.net.apply(init_fun)
        elif initializer is Initializer.FROM_PARAMS:
            self.restore_params(parameters)

    def init(self, inputs, labels, initializer, parameters=None):
        raise NotImplementedError

    def step_init(self, inputs, labels):
        """
        Just in case there needs to be some init function before each step.
        """
        pass

    def step(self, inputs, labels):
        raise NotImplementedError

    def save_params(self):
        return [p.clone() for p in self.net.parameters()]

    def restore_params(self, params):
        with torch.no_grad():
            for old_p, new_p in zip(params, self.net.parameters()):
                new_p.copy_(old_p)

    def jacobian(self, y):
        """
        :param y: Shape (N, c).
        :return: J with shape (N*c, size(params)).
        """
        self.net.zero_grad()

        ysize = y.numel()
        J = np.empty((ysize, self.numparams()))

        for i, yi in enumerate(y):  # Samples.
            for j, c in enumerate(yi):  # Classes.
                # Free the graph at the last backward call.
                if i == y.size(0) - 1 and j == yi.size(0) - 1:
                    c.backward(retain_graph=False)
                else:
                    c.backward(retain_graph=True)

                start_index = 0
                for p in self.net.parameters():
                    J[i * y.size()[1] + j, start_index:start_index + p.numel()] = torch.reshape(p.grad, (-1,))

                    start_index += p.numel()

                self.net.zero_grad()

        return J

    def jacobian_torch(self, y, retain_graph=False):
        """
        :param y: Shape (N, c).
        :param retain_graph: If False, does not retain graph after call.
        :return: J with shape (N*c, size(params)).
        """
        self.net.zero_grad()

        ysize = y.numel()
        J = torch.empty(ysize, self.numparams(), dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        for i, yi in enumerate(y):  # Samples.
            for j, c in enumerate(yi):  # Classes.
                # Free the graph at the last backward call.
                if i == y.size(0) - 1 and j == yi.size(0) - 1:
                    c.backward(retain_graph=retain_graph)
                else:
                    c.backward(retain_graph=True)

                start_index = 0
                for p in self.net.parameters():
                    J[i * y.size()[1] + j, start_index:start_index + p.numel()] = torch.reshape(p.grad, (-1,))

                    start_index += p.numel()

                self.net.zero_grad()

        return J

    def numparams(self):
        return sum(p.numel() for p in self.net.parameters())

    def vec_to_params_update(self, s, from_numpy=True):
        """
        Extracts the network parameters from the vector s, adds them to the current parameters and returns them.

        :param s: Input vector.
        :param from_numpy: If true, s is a numpy vector, else a torch vector.
        :return: List of the torch tensor network parameters.
        """
        param_list = []

        start_index = 0
        for p in self.net.parameters():
            with torch.no_grad():
                if from_numpy:
                    params = torch.from_numpy(s[start_index:start_index + p.numel()])
                else:
                    params = s[start_index:start_index + p.numel()]

                params_rs = torch.reshape(params, p.size())
                param_list.append(p + params_rs)

            start_index += p.numel()

        return param_list

    def zero_grad(self):
        self.net.zero_grad()

    def forward(self, inputs, chunk_size_factor=None, requires_grad=False):
        N = inputs.size(0)

        if chunk_size_factor is None:
            chunk_size_factor = 1

        chunk_size = int(chunk_size_factor * N)

        y = torch.empty((N, self.net.output_dim), dtype=global_config.cfg.datatype, device=global_config.cfg.device)

        for i in range(0, N, chunk_size):
            with torch.set_grad_enabled(requires_grad):
                y[i:i + chunk_size] = self.net(inputs[i:i + chunk_size])

        return y
