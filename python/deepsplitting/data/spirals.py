import pandas as pd
import torch
import torchvision

from deepsplitting.data.misc import get_sampler
from deepsplitting.data.misc import To64fTensor


class BinarySpirals(torch.utils.data.Dataset):
    def __init__(self, folder='datasets/binary_spirals/', transform=None, target_transform=None):
        self.files = {'X_train': folder + 'binary_spirals_X_train',
                      'y_train': folder + 'binary_spirals_y_train'}

        self.X_train = None
        self.y_train = None
        self.transform = transform
        self.target_transform = target_transform

        self.read_csv()

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, item):
        x = self.X_train[item, :]
        y = self.y_train[item, :]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y

    def read_csv(self):
        self.X_train = pd.read_csv(self.files['X_train'], header=None).values.astype('float64').reshape(-1, 2)
        self.y_train = pd.read_csv(self.files['y_train'], header=None).values.astype('float64').reshape(-1, 2)


def load_spirals(training_samples=-1, target_transform=None):
    transform = To64fTensor()

    if target_transform is not None:
        target_transform = torchvision.transforms.Lambda(target_transform)

    dataset = BinarySpirals(transform=transform, target_transform=target_transform)

    training_sampler, training_batch_size_full = get_sampler(training_samples, dataset)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=training_batch_size_full,
                                             shuffle=False, sampler=training_sampler)

    classes = 2

    return dataloader, training_batch_size_full, classes
