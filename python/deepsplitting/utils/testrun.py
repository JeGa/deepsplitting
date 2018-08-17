import torch
import deepsplitting.utils.misc
import deepsplitting.utils.global_config as global_config


def evaluate_nll(outputs, labels):
    sm = torch.nn.Softmax(1)
    _, predicted = torch.max(sm(outputs.data), 1)

    return labels == predicted


def evaluate_ls(outputs, labels, classes):
    _, predicted = torch.max(outputs.data, 1)

    predicted_one_hot = deepsplitting.utils.misc.one_hot(predicted, classes).to(global_config.cfg.device)
    return torch.sum(labels == predicted_one_hot, 1) == labels.size(1)


def run(net, testloader, eval_results):
    results = []

    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs, labels = inputs.to(global_config.cfg.device), labels.to(global_config.cfg.device)

            outputs = net(inputs).detach()

            results.append(eval_results(outputs, labels))

    samples = 0
    correct = 0
    for batch in results:
        correct += batch.sum().item()
        samples += len(batch)

    return correct, samples


def test_nll(net, testloader):
    return run(net, testloader, evaluate_nll)


def test_ls(net, testloader, classes):
    return run(net, testloader, lambda outputs, labels: evaluate_ls(outputs, labels, classes))


def test_at_interval(net, iteration, inputs, labels, classes):
    if global_config.cfg.loss_type == 'ls':
        eval = lambda outputs, labels: evaluate_ls(outputs, labels, classes)
    elif global_config.cfg.loss_type == 'nll':
        eval = evaluate_ls
    else:
        raise ValueError("Unsupported loss type.")

    interval = global_config.cfg.test_interval

    if interval == -1:
        return None

    if iteration % interval != 0:
        return None

    outputs = net(inputs).detach()

    correct = eval(outputs, labels).sum().item()

    return correct


def run_fullbatch_loaded(net, inputs, labels, eval_results):
    results = []

    outputs = net(inputs)

    results.append(eval_results(outputs, labels))

    samples = inputs.size(0)
    # correct =

    # return correct, samples
