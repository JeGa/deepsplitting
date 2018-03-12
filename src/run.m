clearvars;
close all;

addpath(genpath('nn'));

% Classification: spirals, Regression: reg_sinus.
[X_train, y_train, X_test, y_test, dim, classes] = get_data(1, 'spirals', true);

%%

% Define network architecture.
layers = [dim, 10, 10, 5, classes];
[h, dh] = activation_function(2);
%loss = LeastSquares;
loss = NLLSoftmax;

network = Network(layers, h, dh, loss, X_train);

%network = network.check_gradients(layers, X_train, y_train);
network = network.train(X_train, y_train, get_params(2));

[~, y] = network.fp(X_train);
y = Softmax.softmax(y);
plot_result_cls(X_train, y, 3);

[~, y] = network.fp(X_test);
plot_result_cls(X_test, y, 4);

%% Helper functions.

function params = get_params(ls)
    params.linesearch = ls; % 1 = fixed stepsize, 2 = Armijo, 3 = Powell-Wolfe.
    if ls == 1
        params.stepsize = 0.1;
    elseif ls == 2
        params.beta = 0.5;
        params.gamma = 10^-4;
    elseif ls == 3
        params.gamma = 10^-4;
        params.eta = 0.7;
        params.beta = 4;
    else
        error('Unsupported linesearch parameter.');
    end
    
    params.iterations = 500;
    params.plot = 0;
end

function [X_train, y_train, X_test, y_test, dim, classes] = get_data(type, data_type, do_plot)
    addpath('datasets');

    if type == 1
        switch data_type
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
                data = twospirals(1000, 360, 90, 1.2);
        end
    elseif type == 2
        if data_type == 'reg_sinus'
            x = linspace(0, 2*pi, 30);
            y = sin(x) + 0.1 * randn(size(x));
            data = [x; y]';
        end
    else
       error('Unsupported type.'); 
    end

    dim = size(data, 2)-1;

    % Shuffle data.
    shuffle = randsample(1:size(data, 1), size(data, 1));
    X = data(shuffle, 1:dim)';
    
    if type == 2
        % Regression.
        y = data(shuffle, dim+1);
    else
        % Classification.
        y = data(shuffle, dim+1)+1;
    end

    % Divide data into training and test set.
    n = uint64((2/3)*size(data, 1));

    X_train = X(:, 1:n);
    y_train = y(1:n, :);
    X_test = X(:, n+1:end);
    y_test = y(n+1:end, :);
    
    if type == 2
        classes = 1;
        y_train = y_train';
        y_test = y_test';
    else
        % Classification
        y_train = one_hot(y_train);
        y_test = one_hot(y_test);
        classes = max(y);
    end
    
    if do_plot
        if dim == 2
            figure(1);
            scatter(data(:,1), data(:,2), 10, data(:,3));
            axis equal;
            title('Ground truth');
            drawnow
        elseif dim == 1
            figure(1);
            scatter(data(:,1), data(:,2));
            title('Ground truth');
            drawnow 
        end
    end
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

function plot_result_cls(X_train, y, f)
    [~,C] = max(y, [], 1);
    C = C - 1;

    % y is of shape (cls, samples).
    figure(f);
    scatter(X_train(1,:), X_train(2,:), 10, C);
    axis equal;
    drawnow
end

function plot_result_reg(X_train, y, f)
    figure(f);
    scatter(X_train, y);
    drawnow
end

function x_onehot = one_hot(x)
    % x: (N,1).
    classes = max(x);

    x_onehot = zeros(classes, size(x,1));
    ind = sub2ind(size(x_onehot), x', 1:size(x',2));
    x_onehot(ind) = 1;
end