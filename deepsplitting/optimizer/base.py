import torch
import numpy as np


class Hyperparams:
    def __init__(self, **params):
        self.__dict__.update(params)


class BaseOptimizer:
    def __init__(self, net, hyperparams, debug=False):
        self.net = net
        self.hyperparams = hyperparams

        def init(submodule):
            if type(submodule) == torch.nn.Linear:
                if debug:
                    submodule.weight.data.fill_(1)
                    submodule.bias.data.fill_(1)
                else:
                    torch.nn.init.normal_(submodule.weight)
                    torch.nn.init.normal_(submodule.bias)

                    submodule.weight.data.mul_(0.1)
                    submodule.bias.data.mul_(0.1)

        net.apply(init)

    def step(self, inputs, labels):
        raise NotImplementedError

    def eval_print(self, inputs, labels):
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
                c.backward(retain_graph=True)

                start_index = 0
                for p in self.net.parameters():
                    J[i * y.size()[1] + j, start_index:start_index + p.numel()] = torch.reshape(p.grad, (-1,))

                    start_index += p.numel()

                self.net.zero_grad()

        return J

    def numparams(self):
        return sum(p.numel() for p in self.net.parameters())

    def vec_to_params_update(self, s):
        """
        Extracts the network parameters from the numpy vector s, adds them to the current parameters and returns them.

        :param s: Numpy vector.
        :return: List of the torch tensor network parameters.
        """
        param_list = []

        start_index = 0
        for p in self.net.parameters():
            with torch.no_grad():
                params = torch.from_numpy(s[start_index:start_index + p.numel()])
                params_rs = torch.reshape(params, p.size())
                param_list.append(p + params_rs)

            start_index += p.numel()

        return param_list

    def zero_grad(self):
        self.net.zero_grad()
