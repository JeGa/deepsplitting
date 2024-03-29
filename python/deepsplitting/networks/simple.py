import torch
import torch.nn.functional as F
from .base import BaseNetwork


class SimpleConvNet_mnist(BaseNetwork):
    def __init__(self, h, criterion):
        super(SimpleConvNet_mnist, self).__init__()

        self.output_dim = 10

        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.fc1 = torch.nn.Linear(320, 50)
        self.fc2 = torch.nn.Linear(50, 10)

        self.h = h
        self.criterion = criterion

    def forward(self, x):
        x = self.h(F.max_pool2d(self.conv1(x), 2))
        x = self.h(F.max_pool2d(self.conv2(x), 2))

        x = x.view(-1, 320)

        x = self.h(self.fc1(x))
        x = self.fc2(x)

        return x


class SimpleConvNet_cifar(BaseNetwork):
    """
    From pytorch tutorial.

    Input: (3,32,32) image.
    Output: 10 classes.
    """

    def __init__(self, h, criterion):
        super(SimpleConvNet_cifar, self).__init__()

        self.output_dim = 10

        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, self.output_dim)

        self.h = h
        self.criterion = criterion

    def forward(self, x):
        x = self.pool(self.h(self.conv1(x)))
        x = self.pool(self.h(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = self.h(self.fc1(x))
        x = self.h(self.fc2(x))
        x = self.fc3(x)

        return x


class SimpleSmallConvNet(BaseNetwork):
    """
    From pytorch tutorial.

    Input: (3,32,32) image.
    Output: 10 classes.
    """

    def __init__(self, h, criterion):
        super(SimpleSmallConvNet, self).__init__()

        self.output_dim = 10

        self.conv1 = torch.nn.Conv2d(3, 3, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)

        self.fc1 = torch.nn.Linear(3 * 14 * 14, 42)
        self.fc2 = torch.nn.Linear(42, self.output_dim)

        self.h = h
        self.criterion = criterion

    def forward(self, x):
        x = self.pool(self.h(self.conv1(x)))

        x = x.view(-1, 3 * 14 * 14)

        x = self.h(self.fc1(x))
        x = self.fc2(x)

        return x


class SimpleFFNet(BaseNetwork):
    def __init__(self, layers, h, criterion):
        super(SimpleFFNet, self).__init__()

        self.fclayers = []

        for i in range(len(layers) - 2):
            submodule = torch.nn.Linear(layers[i], layers[i + 1])

            self.fclayers.append(submodule)
            self.add_module('fc' + str(i), submodule)

        self.linear_layer = torch.nn.Linear(layers[-2], layers[-1])

        self.output_dim = layers[-1]

        self.h = h
        self.criterion = criterion
        self.layer_size = layers

        self.z = []
        self.a = []

    def forward(self, x):
        self.z = []
        self.a = []

        for i in range(len(self.fclayers)):
            if i == 0:
                zi = self.fclayers[i](x)
            else:
                zi = self.fclayers[i](self.a[-1])
            ai = self.h(zi)

            self.z.append(zi)
            self.a.append(ai)

        zL = self.linear_layer(self.a[-1])
        self.z.append(zL)

        return zL

    def weights(self, L):
        """
        Returns the weight matrix of layer L.
        """
        params = self.parameters()

        return list(params)[L + L]

    def bias(self, L):
        """
        Returns the bias vector of layer L.
        """
        params = self.parameters()

        return list(params)[L + L + 1]

    def set_weights(self, L, new_weight):
        with torch.no_grad():
            self.weights(L).copy_(new_weight)

    def set_bias(self, L, new_bias):
        with torch.no_grad():
            self.bias(L).copy_(new_bias)
