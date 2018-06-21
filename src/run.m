clearvars;
close all;
rng(123)

addpath(genpath('nn'));
addpath(genpath('baseline'));

% Classification: spirals, Regression: reg_sinus.
[X_train, y_train, X_test, y_test, dim, classes] = helper.get_data(1, 'spirals', false, 1);

%%

activation_type = 2; % 1 = sigmoid, 2 = relu.
loss_type = 2; % 1 = LS, 2 = NLLsm.

% Define network architecture.
layers = [dim, 12, 12, 12, classes];
[h, dh] = helper.activation_function(activation_type);

N = size(y_train, 2);

loss = get_loss(loss_type);

networks = {
    GDNetwork(layers, h, dh, loss, X_train), 'GD';
    LLCNetwork(layers, h, dh, loss, X_train), 'LLC';
    ProxDescentNetwork(layers, h, dh, loss, X_train), 'ProxDescent';
    ProxPropNetwork(layers, h, dh, loss, X_train), 'ProxProp'; % TODO: Armijo.
};

if loss_type == 1
   networks(end,:) = {LMNetwork(layers, h, dh, loss, X_train), 'LM'};
end

train(networks{4,1}, X_train, y_train, networks{4,2}, loss_type, activation_type);
%train_all(networks, X_train, y_train, loss_type, activation_type);

results = load_results(networks);

%plot_grid(network);

function train_all(networks, X_train, y_train, loss_type, activation_type)
    types = {networks{:,2}};
    
    n1 = networks{1,1};
    [W, b] = n1.get_params();
    
    for i = 1:size(types, 2)
        ni = networks{i,1};
        networks{i,1} = ni.set_params(W, b);
        train(networks{i,1}, X_train, y_train, networks{i,2}, loss_type, activation_type);
    end
end

function results = load_results(networks)
    types = {networks{:,2}};

    folder = 'results/';

    results = cell(1, size(types, 2));

    figure;
    title('Spiral data set.');
    xlabel('Iteration');
    ylabel('Objective');
    
    for i = 1:size(types, 2)
        res = load(join([folder, 'losses_', types{i}, '.mat']));
        results{i} = res.losses;
        
        loss_type = getLossFromType(res.loss_type);
        activation_type = getActivationFromType(res.activation_type);
        
        hold on
        plot(res.losses, 'DisplayName', ...
            join([types{i}, '(', num2str(res.time), 's, ', loss_type, ', ', activation_type, ')']));
    end
    
    hold off
    legend
end

function [losses, misclassified] = train(network, X_train, y_train, type, loss_type, activation_type)
    folder = 'results/';
    
    tic;
    [network, losses] = network.train(X_train, y_train, helper.get_params(type));
    time = toc;
    
    if loss_type == 1
        y = helper.predict_ls(network, X_train);
    elseif loss_type == 2
        y = helper.predict_nllsm(network, X_train);
    else
       error('Unknown loss type.')
    end
    
    misclassified = helper.results_cls(y, y_train);

    save(join([folder, 'losses_', type, '.mat']), 'losses', 'misclassified', 'time', 'loss_type', 'activation_type');
end

function loss = get_loss(type)
    if type == 1
        loss = LeastSquares(1);
        % TODO: N scaling factor... % TODO: Bad with 1/N scaling and LS loss.
    elseif type == 2
        loss = NLLSoftmax();
    else
        error('Unknown loss type.')
    end
end

function t = getLossFromType(loss_type)
    if loss_type == 1
        t = 'Least Squares';
    elseif loss_type == 2
        t = 'NLL';
    else
        warning('Unsupported loss type.');
    end
end

function t = getActivationFromType(activation_type)
    if activation_type == 1
        t = 'sigmoid';
    elseif activation_type == 2
        t = 'relu';
    else
        warning('Unsupported activation type');
    end
end

function check_gradient()
    %network = network.check_gradients(layers, X_train, y_train);
    %network = network.check_jacobian(layers, X_train, y_train);
    %network = network.check_gradients_primal2(layers, X_train, y_train);
end