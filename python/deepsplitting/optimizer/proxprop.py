import torch
from torch.nn import functional as F
import scipy

from .base import BaseOptimizer
import deepsplitting.optimizer.misc as misc


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams):
        super().__init__(net, hyperparams)

        if self.net.h is F.relu:
            self.h_grad = misc.grad_relu
        elif self.net.h is F.sigmoid:
            self.h_grad = misc.grad_sigmoid
        else:
            raise ValueError("Acivation not supported.")

    def init_variables(self, inputs, labels):
        # Initialize non-linearities a and linearities z.
        self.net(inputs)

    def init(self, inputs, labels, initializer, parameters=None):
        super(Optimizer, self).init_parameters(initializer, parameters)

        self.init_variables(inputs, labels)

    def step(self, inputs, labels):
        y = self.net(inputs).detach()
        y.requires_grad_()

        # With last linear layer.
        L = len(self.net.layer_size) - 2
        N = inputs.size(0)
        tau = self.hyperparams.tau
        tau_theta = self.hyperparams.tau_theta

        L_current = self.net.criterion(y, labels)
        L_current.backward()
        loss_grad = y.grad.data

        a = self.net.a
        z = self.net.z

        # Last layer.
        a[L - 1] = a[L - 1] - tau * (loss_grad.matmul(self.net.weights(L)))

        self.net.set_weights(L, self.net.weights(L) - tau * (loss_grad.t().matmul(a[L - 1])))
        self.net.set_bias(L, self.net.bias(L) - tau * (torch.sum(loss_grad, 0)))

        # Other layers.
        for i in range(L - 1, 0, -1):
            z_save = torch.tensor(z[i])
            z[i] = z[i] - self.h_grad(z[i]).mul(self.net.h(z[i]) - a[i])

            if i != 0:
                # Save before update.
                a_save = torch.tensor(a[i - 1])

                # Gradient step.
                a[i - 1] = a[i - 1] - (z_save - z[i]).matmul(self.net.weights(i))
            else:
                a_save = inputs

            # Prox operator.
            a_aug = torch.cat((a_save.t(), torch.ones(1, N, dtype=torch.float)), 0)

            A = a_aug.matmul(a_aug.t()) + (1 / tau_theta) * torch.eye(a_aug.size(0), dtype=torch.float)
            B = z[i].t().matmul(a_aug.t()) + (1 / tau_theta) * torch.cat(
                (self.net.weights(i), torch.unsqueeze(self.net.bias(i), 1)), 1)

            A = A.detach().numpy()
            B = B.detach().numpy()

            theta = scipy.linalg.solve(A.T, B.T).T

            self.net.set_weights(i, torch.from_numpy(theta[:, 0:-1]))
            self.net.set_bias(i, torch.from_numpy(theta[:, -1]).squeeze())

        return L_current.item(), self.net.loss(inputs, labels).item()
