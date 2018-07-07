import torch
import numpy as np


class Hyperparams:
    def __init__(self, **params):
        self.__dict__.update(params)


class BaseOptimizer:
    def __init__(self, net, hyperparams):
        self.net = net
        self.hyperparams = hyperparams

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

        for i, yi in enumerate(y):  # Samples
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
