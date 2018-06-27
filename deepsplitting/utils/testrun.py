import torch
import logging
import deepsplitting


def run(net, testloader, eval_results):
    results = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data

            outputs = net(inputs)

            results.append(eval_results(outputs, labels))

    samples = 0
    correct = 0
    for batch in results:
        correct += batch.sum().item()
        samples += len(batch)

    logging.info("{} of {} correctly classified.".format(correct, samples))


def test_softmax(net, testloader):
    def evaluate(outputs, labels):
        sm = torch.nn.Softmax(1)
        _, predicted = torch.max(sm(outputs.data), 1)

        return labels == predicted

    run(net, testloader, evaluate)


def test_scores(net, testloader):
    def evaluate(outputs, labels):
        _, predicted = torch.max(outputs.data, 1)

        predicted_one_hot = deepsplitting.utils.misc.one_hot(predicted, 10)
        return torch.sum(labels == predicted_one_hot, 1) == labels.size(1)

    run(net, testloader, evaluate)
