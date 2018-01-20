function [W, b, z, a, lambda] = nn(layers, n, h, X_train)

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
    
end

