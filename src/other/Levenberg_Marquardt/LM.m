rng(1);
clear all;
clc
close all;

N = 2000;

noise_var = 0.01;
% generate training data
X = rand(1, N)*2*pi;
Y = sin(X) + noise_var*randn(size(X));;


% Define network architecture. Number of neurons per layer. Each layer
% fully connected.
layers = [size(X, 1), 12, 12, size(Y, 1)];

% Number of layers (including last linear layer).
L = size(layers, 2)-1;

max_iter = 100;

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


factor = 10;
mu = 0.001;
for it = 1:max_iter
    [obj, J, v] = backprop(x, layers, f, df, ell, dell, X, Y);
    loop = true;
    while(loop)
        step = (J'*J+mu*eye(size(x,1)))\(J'*v);
        if(obj < loss(x-step, layers, f, ell, X, Y))
            mu = mu*factor;
        else
            mu = mu/factor;
            loop = false;
        end
    end
    x = x - step;
    
    fprintf('it=%d  loss=%f  mu=%d\n', it, loss(x, layers, f, ell, X, Y), mu);
    if(abs(obj-loss(x-step,layers,f,ell,X,Y))<1e-12)
        break
    end


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
