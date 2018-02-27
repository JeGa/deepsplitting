clearvars;
close all;

addpath('datasets');

%data = corners();
%data = outlier();
%data = halfkernel();
data = twospirals(5000, 560, 90, 1.2);
%data = crescentfullmoon();
%data = clusterincluster();

scatter(data(:,1), data(:,2), 10, data(:,3));
axis equal;
title('Ground truth');
drawnow

% Shuffle data.
shuffle = randsample(1:size(data, 1), size(data, 1));
X = data(shuffle, 1:2)';
y = data(shuffle, 3)+1;

%%

% Define network architecture.
layers = [size(X, 1), 10, 10, 5, max(y)];

% Number of layers (including last linear layer).
L = size(layers, 2)-1;

max_iter = 500;

% Weights
W = cell(1, L);
% Biases
b = cell(1, L);

% Initialize network weights.
for l=1:L
    W{l} = 0.1*randn(layers(l+1), layers(l));
    b{l} = 0.1*randn(layers(l+1), 1);
end


% Use sigmoid activation function.
f = @(t) 1 ./ (1 + exp(-t));
df = @()


for i=1:max_iter
    % TODO
end

