function [a, z] = model(X, W, b, L, f)
    % Variables for linearities.
    z = cell(1, L);
    % Variables for activations.
    a = cell(1, L-1);

    N = size(X, 2);
    
    % Perform a forward pass to find a feasible initialization of all variables.
    z{1} = W{1}*X + b{1} * ones(1, N);

     for l=1:L-1
        % Apply activation function.
        a{l} = f(z{l});

        % Apply linear mapping.
        z{l+1} = W{l+1}*a{l} + b{l+1} * ones(1, N);
    end
end

