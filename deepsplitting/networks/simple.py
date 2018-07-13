import torch


class SimpleConvNet(torch.nn.Module):
    """
    From pytorch tutorial.

    Input: (3,32,32) image.
    Output: 10 classes.
    """

    def __init__(self, h, criterion):
        super(SimpleConvNet, self).__init__()

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

    def loss(self, inputs, labels):
        return self.criterion(self(inputs), labels)


class SimpleSmallConvNet(torch.nn.Module):
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

    def loss(self, inputs, labels):
        return self.criterion(self(inputs), labels)


class SimpleFFNet(torch.nn.Module):
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

    def forward(self, x):
        for i in range(len(self.fclayers)):
            x = self.h(self.fclayers[i](x))

        x = self.linear_layer(x)

        return x

    def loss(self, inputs, labels):
        return self.criterion(self(inputs), labels)
