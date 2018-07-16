import torch
from torch.nn import functional as F
import scipy

from .base import BaseOptimizer


class Optimizer(BaseOptimizer):
    def __init__(self, net, hyperparams):
        super().__init__(net, hyperparams)

        if self.net.h is F.relu:
            self.h_grad = self.h_grad_relu
        elif self.net.h is F.sigmoid:
            self.h_grad = self.h_grad_sigmoid
        else:
            raise ValueError("Acivation not supported.")

    def init_variables(self, inputs, labels):
        self.net(inputs)

    def init(self, inputs, labels, debug=False):
        super(Optimizer, self).init_parameters(debug)

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
        # TODO for cnn: weights(L) = d/da phi(a) diagonal part.
        a[L - 1] = a[L - 1] - tau * (loss_grad.matmul(self.weights(L)))

        self.set_weights(L, self.weights(L) - tau * (loss_grad.t().matmul(a[L - 1])))
        self.set_bias(L, self.bias(L) - tau * (torch.sum(loss_grad, 0)))

        # Other layers.
        for i in range(L - 1, 0, -1):
            z[i] = z[i] - self.h_grad(z[i]).mul(self.net.h(z[i]) - a[i])

            if i != 0:
                # Save before update.
                a_save = torch.tensor(a[i - 1])

                # Forward pass with new z.
                yi = a[i - 1].matmul(self.weights(i).t()) + (
                    self.bias(i).mul(torch.ones((N, 1), dtype=torch.double))) - z[i]

                # Gradient step.
                a[i - 1] = a[i - 1] - yi.matmul(self.weights(i))
            else:
                a_save = inputs

            # Prox operator.
            a_aug = torch.cat((a_save.t(), torch.ones(1, N, dtype=torch.double)), 0)

            A = a_aug.matmul(a_aug.t()) + (1 / tau_theta) * torch.eye(a_aug.size(0), dtype=torch.double)
            B = z[i].t().matmul(a_aug.t()) + (1 / tau_theta) * torch.cat(
                (self.weights(i), torch.unsqueeze(self.bias(i), 1)), 1)

            A = A.detach().numpy()
            B = B.detach().numpy()

            theta = scipy.linalg.solve(A.T, B.T).T

            self.set_weights(i, torch.from_numpy(theta[:, 0:-1]))
            self.set_bias(i, torch.from_numpy(theta[:, -1]).squeeze())

        return L_current.item(), self.net.loss(inputs, labels).item()

    def weights(self, L):
        """
        Returns the weight matrix of layer L.
        """
        params = self.net.parameters()

        return list(params)[L + L]

    def bias(self, L):
        """
        Returns the bias vector of layer L.
        """
        params = self.net.parameters()

        return list(params)[L + L + 1]

    def set_weights(self, L, new_weight):
        with torch.no_grad():
            self.weights(L).copy_(new_weight)

    def set_bias(self, L, new_bias):
        with torch.no_grad():
            self.bias(L).copy_(new_bias)

    def h_autograd(self, x):
        """
        Gradient of the activation function. This should work for all activation functions, so autograd is used here.
        """
        x.retain_grad()
        y = self.net.h(x)

        g = torch.empty(x.size(), dtype=torch.double)

        for i, yi in enumerate(y):  # Samples.
            for j, c in enumerate(yi):  # Classes.
                c.backward(retain_graph=True)
                # Everything except one entry is zero.
                g[i, j] = torch.sum(x.grad)
                self.zero_grad_tensor(x)

        return g

    def h_grad_relu(self, x):
        """
        :param x: (N, d)
        :returns: J in matrix form (N, d).
        """
        return F.sigmoid(x).mul(1 - F.sigmoid(x))

    def h_grad_sigmoid(self, x):
        """
        :param x: (N, d)
        :returns: J in matrix form (N, d).
        """
        raise NotImplementedError
        # return x > 0

    def zero_grad_tensor(self, t):
        # TODO: t.grad.detach_()
        t.grad.zero_()

    def J_phi(self, L):
        """
        Jacobian of the linear transfer function of the neural network.
        Returns the matrix version.

        phi(x) = phi(x;A,B).

        In matrix form:
            x: (d, N)
            y: (c, N)

        phi(x) = A * x + B = A * x + [b,b,...,b]

        In vector form:
            x: (d*N)
            y: (c*N)

        phi(vec(x)) = Ad * vec(x) + vec(B)

        The jacobians are:

        J_vec = Ad = diag(A)
        J = A
        """
        return self.weights(L)
