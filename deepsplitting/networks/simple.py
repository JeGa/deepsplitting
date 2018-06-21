import torch.nn as nn


class SimpleNet(nn.Module):
    """
    From pytorch tutorial.

    Input: (3,32,32) image.
    Output: 10 classes.
    """

    def __init__(self, h):
        super(SimpleNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.h = h

    def forward(self, x):
        x = self.pool(self.h(self.conv1(x)))
        x = self.pool(self.h(self.conv2(x)))

        x = x.view(-1, 16 * 5 * 5)

        x = self.h(self.fc1(x))
        x = self.h(self.fc2(x))
        x = self.fc3(x)

        return x
