function [val] = loss(x, layers, f, ell, X, Y)
%LOSS Summary of this function goes here
%   Detailed explanation goes here
% Number of layers (including last linear layer).
    L = size(layers, 2)-1;
    
    % Weights
    W = cell(1, L);
    % Biases
    b = cell(1, L);
    
    
    % Initialize network weights.
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
    
    
    [~, z] = model(X, W, b, L, f);
    val = ell(z{L}, Y);
end

