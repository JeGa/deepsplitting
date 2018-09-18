import torch
import torch.nn.functional as F
from .base import BaseNetwork

import torch.nn as nn


class Autoencoder_mnist(BaseNetwork):
    def __init__(self, h, criterion):
        super(Autoencoder_mnist, self).__init__()

        self.output_dim = 28 * 28

        self.criterion = criterion

        if h == F.relu:
            activation = torch.nn.ReLU
        elif h == F.sigmoid:
            activation = torch.nn.Sigmoid
        else:
            raise ValueError("Unsupported activation function")

        self.encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, 16, 3, stride=1, padding=1),  # 28x28
            nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 14x14

            torch.nn.Conv2d(16, 8, 3, stride=1, padding=1),  # 14x14
            nn.ReLU(True),
            torch.nn.MaxPool2d(2),  # 7x7

            torch.nn.Conv2d(8, 8, 3, stride=1, padding=2),  # 7x7
            nn.ReLU(True),
            torch.nn.MaxPool2d(2)  # 4x4
        )

        # Encoder output is (N, 8, 4, 4).

        self.decoder = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(8, 8, 3, stride=2, padding=1),
            nn.ReLU(True),
            # torch.nn.Upsample(scale_factor=2, mode='bilinear'),

            torch.nn.ConvTranspose2d(8, 8, 3, stride=1, padding=1),
            nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),

            torch.nn.ConvTranspose2d(8, 16, 3, stride=1, padding=1),
            nn.ReLU(True),
            torch.nn.Upsample(scale_factor=2, mode='nearest'),

            torch.nn.ConvTranspose2d(16, 1, 3, stride=1, padding=1),
            torch.nn.Tanh()
        )

        # self.encoder = nn.Sequential(
        #     nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
        #     nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
        #     nn.ReLU(True),
        #     nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        # )
        # self.decoder = nn.Sequential(
        #     nn.ConvTranspose2d(8, 16, 3, stride=2),  # b, 16, 5, 5
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
        #     nn.ReLU(True),
        #     nn.ConvTranspose2d(8, 1, 2, stride=2, padding=1),  # b, 1, 28, 28
        #     nn.Tanh()
        # )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0), -1)
