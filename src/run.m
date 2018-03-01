clearvars;
close all;

addpath(genpath('nn'));

[X_train, y_train, X_test, y_test, dim, classes] = get_data('spirals', false);

%%

% Define network architecture.
layers = [dim, 10, 10, 5, classes];
[h, dh] = activation_function('sigmoid');
loss = LeastSquares;

% Initialize variables.
network = Network(layers, h, loss, X_train);

nezwork = network.train(X_train, y_train);

[~, y] = network.fp(X_train);

plot_result(X_train, y);

%% Helper functions.

function [X_train, y_train, X_test, y_test, dim, classes] = get_data(type, plot)
    addpath('datasets');

    switch type
        case 'corners'
            data = corners();
        case 'outlier'
            data = outlier();
        case 'halfkernel'
            data = halfkernel();
        case 'moon'
            data = crescentfullmoon();
        case 'clusters'
            data = clusterincluster();
        case 'spirals'
            data = twospirals(5000, 560, 90, 1.2);
    end

    if plot
        scatter(data(:,1), data(:,2), 10, data(:,3));
        axis equal;
        title('Ground truth');
        drawnow
    end

    % Shuffle data.
    shuffle = randsample(1:size(data, 1), size(data, 1));
    X = data(shuffle, 1:2)';
    y = data(shuffle, 3)+1;

    dim = size(X, 1);
    classes = max(y);

    % Divide data into training and test set.
    n = uint64((2/3)*size(data, 1));

    X_train = X(:, 1:n);
    y_train = y(1:n, :);
    X_test = X(:, n+1:end);
    y_test = y(n+1:end, :);
end

function [h, dh] = activation_function(type)
    if type == 'sigmoid'
        h = @(t) 1 ./ (1 + exp(-t));
        dh = @(t) h(t) .* (1 - h(t));
    elseif type == 'relu'
        h = @(t) max(0, t);
        dh = @(t) 1 * (t>0);
    end
end

function plot_result(X_train, y)
    % y is of shape (cls, samples).
    scatter(X_train(1,:), X_train(2,:), 10, max(y, [], 1));
    axis equal;
    title('Ground truth');
    drawnow
end
