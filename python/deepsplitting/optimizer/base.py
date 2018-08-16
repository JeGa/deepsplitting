import math

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

    def loss_chunked(self, inputs, labels):
        y = self.forward(inputs, global_config.cfg.forward_chunk_size_factor)
        return self.net.criterion(y, labels).item()

    def load(self, params):
        if params is not None:
            saved_params = self.save_params()
            self.restore_params(params)
        else:
            saved_params = None
        return saved_params

    def restore(self, saved_params):
        if saved_params is not None:
            self.restore_params(saved_params)

    def subsampled_jacobians(self, index, subindex, inputs):
        y_batch = self.forward(inputs[index], requires_grad=True)

        J1 = self.jacobian_torch(y_batch).detach()

        y_subsampled = y_batch[subindex]

        c = y_subsampled.size(1)
        subsampled_indices = [item for j in subindex for item in range(j * c, j * c + c)]

        J2 = J1[subsampled_indices, :]

        return J1, J2, y_batch.detach(), y_subsampled.detach()

    def cg_solve(self, x, A, B, return_step=False):
        """
        Solve for: Ax = B with linear mapping A.

        The argument A should be a function. It is often better to calculate e.g. Ax = A1*A2*x using matrix vector
        products instead of calculating A directly.
        """
        r = B - A(x)
        p = r

        for i in range(self.hyperparams.cg_iter):
            alpha = (r.t().matmul(r)) / (p.t().matmul(A(p)))

            x = x + alpha * p
            r_new = r - alpha * A(p)

            beta = (r_new.t().matmul(r_new)) / (r.t().matmul(r))
            p = r_new + beta * p

            r = r_new

        if return_step:
            return self.vec_to_params_update(x, from_numpy=False), x
        else:
            return self.vec_to_params_update(x, from_numpy=False)

    @staticmethod
    def fullbatch_subindex_init(batch_size, subsample_factor, N):
        subsample_size = int(subsample_factor * batch_size)

        indices = torch.randperm(N)

        max_steps = int(math.ceil(len(indices) / batch_size))

        return indices, subsample_size, max_steps

    @staticmethod
    def rand_subindex(batch_size, subsample_size):
        return torch.randperm(batch_size)[0:subsample_size]

    @staticmethod
    def batches(indices, batch_size):
        return (indices[i:i + batch_size] for i in range(0, indices.size(0), batch_size))
