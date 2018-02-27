rng(1);
clear all;
clc
close all;

N = 1000;

% generate training data
X = rand(1, N)*2*pi;
Y = sin(X);


% Define network architecture. Number of neurons per layer. Each layer
% fully connected.
layers = [size(X, 1), 12, 12, size(Y, 1)];

% Number of layers (including last linear layer).
L = size(layers, 2)-1;

max_iter = 4000;

% Weights
W = cell(1, L);
% Biases
b = cell(1, L);

% Initialize network weights.
for l=1:L
    W{l} = randn(layers(l+1), layers(l));
    b{l} = randn(layers(l+1), 1);
end

x = zeros(0, 1);
for l=1:L
    x = cat(1, x, W{l}(:));
end

for l=1:L
    x = cat(1, x, b{l});
end

%f = @(x) x;
%df = @(x) ones(size(x));
f = @(x) 1.0 ./ (1.0 + exp(-x));  % Sigmoid function
df = @(x) f(x) .* (1 - f(x)); % derivative

ell = @(z, Y) 0.5*sum(sum((z - Y).^2)); % error function
dell =@(z, Y) z - Y;                    % derivative



%options = optimoptions(@fminunc,'CheckGradients',true,'SpecifyObjectiveGradient',true);
%x = fminunc(@(x) backprop(x, layers, f, df, ell, dell, X, Y), x, options);

 

alpha = 0.01;
beta = 0.3;
for it=1:max_iter    
    [obj, grad] = backprop(x, layers, f, df, ell, dell, X, Y);
   

    tau = 1;
    while(loss(x - tau*grad, layers, f, ell, X, Y) > loss(x, layers, f, ell, X, Y) - alpha*tau*(grad'*grad))
        tau = beta * tau;
    end

    x = x - tau*grad;

    fprintf('it=%d  learningrate=%f  loss=%f  gap=%f\n', it, tau, loss(x, layers, f, ell, X, Y), norm(grad));
end

%%
% Read off weights from solution x.
offset = 0;
for l=1:L
    num_elem = layers(l+1)*layers(l);
    W{l} = reshape(x((1:num_elem)+offset), layers(l+1), layers(l));

    offset = offset + num_elem;
end

for l=1:L
    num_elem = layers(l+1);
    b{l} = x((1:num_elem)+offset);

    offset = offset+layers(l+1);
end

% Evaluate trained model on test data:
X_test = rand(1, 100)*2*pi;
[~, z] = model(X_test, W, b, L, f);

scatter(X_test, z{L});
