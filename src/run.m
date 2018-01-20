clearvars;
close all;

addpath('datasets');

%data = corners();
%data = outlier();
%data = halfkernel();
%data = crescentfullmoon();
%data = clusterincluster();
data = twospirals(5000, 560, 90, 1.2);

%scatter(data(:,1), data(:,2), 10, data(:,3));
%axis equal;
%title('Ground truth');
%drawnow

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

%%

% Define network architecture.
layers = [dim, 10, 10, 5, classes];

% Number of layers (including last linear layer).
L = size(layers, 2)-1;

% Weights
W = cell(1, L);
% Biases
b = cell(1, L);
% Variables for linearities.
z = cell(1, L);
% Variables for activations.
a = cell(1, L-1);
% Lagrange multipliers.
lambda = cell(1, L);

% Initialize network weights.
for l=1:L
    W{l} = 0.1*randn(layers(l+1), layers(l));
    b{l} = 0.1*randn(layers(l+1), 1);
end

% Use sigmoid activation function.
h = @(t) 1 ./ (1 + exp(-t));
dh = @(t) h(t) .* (1 - h(t));

% Perform a forward pass to find a feasible initialization of all variables.
z{1} = W{1}*X_train + b{1} * ones(1, n);

for l=1:L-1
    % Apply activation function.
    a{l} = h(z{l});

    % Apply linear mapping.
    z{l+1} = W{l+1}*a{l} + b{l+1} * ones(1, size(a{l}, 2));
end

% Initialize Lagrange multipliers for equality constraints.
for l=1:L
   lambda{l} = ones(layers(l+1), n); 
end

%%

max_iter = 500;

for i=1:max_iter
    deepsplitting(L, W, a, z, lambda, X_train, y_train, h, dh);
    disp(['Iteration ', num2str(i)]);
end

