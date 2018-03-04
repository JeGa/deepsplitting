clearvars;
close all;

addpath(genpath('nn'));

[X_train, y_train, X_test, y_test, dim, classes] = get_data('spirals', false);

%%

% Define network architecture.
layers = [dim, 20, 20, 10, 5, classes];
[h, dh] = activation_function(2);
loss = LeastSquares;

network = Network(layers, h, dh, loss, X_train);
network = network.train(X_train, y_train, get_params(1));

[~, y] = network.fp(X_train);
plot_result(X_train, y, 3);

%% Helper functions.

function [params] = get_params(ls)
    params.linesearch = ls; % 0 = fixed stepsize, 1 = Armijo, 2 = Wolfe-Powell.
    if ls == 0
        params.stepsize = 0.1;
    elseif ls == 1
        params.beta = 0.5;
        params.gamma = 10^-4;
    elseif ls == 2
        
    end
    
    params.iterations = 2500;
    params.plot = 0;
end

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
            data = twospirals(300, 560, 90, 1.2);
    end

    if plot
        figure(1);
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
    
    y_train = one_hot(y_train);
    y_test = one_hot(y_test);
end

function [h, dh] = activation_function(type)
    if type == 1
        h = @(t) 1 ./ (1 + exp(-t));
        dh = @(t) h(t) .* (1 - h(t));
    elseif type == 2
        h = @(t) max(0, t);
        dh = @(t) 1 * (t>0);
    end
end

function plot_result(X_train, y, f)
    [~,C] = max(y, [], 1);
    C = C - 1;

    % y is of shape (cls, samples).
    figure(f);
    scatter(X_train(1,:), X_train(2,:), 10, C);
    axis equal;
    drawnow
end

function [x_onehot] = one_hot(x)
    % x: (N,1).
    classes = max(x);

    x_onehot = zeros(classes, size(x,1));
    ind = sub2ind(size(x_onehot), x', 1:size(x',2));
    x_onehot(ind) = 1;
end