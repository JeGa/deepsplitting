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

% Use sigmoid activation function.
h = @(t) 1 ./ (1 + exp(-t));
dh = @(t) h(t) .* (1 - h(t));

% Initialize variables.
[W, b, z, a, lambda] = nn(layers, n, h, X_train);

%%

max_iter = 500;

for i=1:max_iter
    deepsplitting(L, W, b, z, a, lambda, X_train, y_train, h, dh);
    disp(['Iteration ', num2str(i)]);
end

