import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

import deepsplitting.data
import deepsplitting.networks.simple
import deepsplitting.util
import deepsplitting.optimizer.GradientDescentArmijo as GD


def test(net, testloader):
    result = []

    with torch.no_grad():
        for data in testloader:
            images, labels = data

            outputs = net(images)

            _, predicted = torch.max(outputs.data, 1)

            result.append(labels == predicted)

    print("{} of {} correctly classified.".format(sum([r.sum().item() for r in result]), sum([len(r) for r in result])))


def train(net, trainloader, optimizer, criterion, epochs):
    losses = list()

    for epoch in range(epochs):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data

            net.zero_grad()

            outputs = net(inputs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step(lambda: criterion(net(inputs), labels))

            losses.append(loss.item())

        if epoch % 10 == 9:
            print('[%d/%d] loss: %.3f' % (epoch + 1, epochs, loss.item()))

    return losses


def main():
    trainloader, testloader, classes = deepsplitting.data.load_data(16, 8)

    deepsplitting.util.show(trainloader)

    net = deepsplitting.networks.simple.SimpleNet(F.relu)

    optimizer = GD.Optimizer(net)

    losses = train(net, trainloader, optimizer, nn.CrossEntropyLoss(), 150)

    plt.figure()
    plt.plot(losses)
    plt.show()

    test(net, trainloader)


if __name__ == '__main__':
    main()
