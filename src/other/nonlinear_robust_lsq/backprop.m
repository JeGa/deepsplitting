function [obj, grad] = backprop(x, layers, f, df, ell, dell, X, Y)
    % Number of layers (including last linear layer).
    N = size(X, 2);
    L = size(layers, 2)-1;
    
    % Weights
    W = cell(1, L);
    % Biases
    b = cell(1, L);
    
    grad_W = cell(1, L);
    grad_b = cell(1, L);
    
    % Read off network weights.
    offset = 0;
    for l=1:L
        num_elem = layers(l+1)*layers(l);
        W{l} = reshape(x((1:num_elem)+offset), layers(l+1), layers(l));
        grad_W{l} = zeros(size(W{l}));
        
        offset = offset + num_elem;
    end
    
    for l=1:L
        num_elem = layers(l+1);
        b{l} = x((1:num_elem)+offset);
        grad_b{l} = zeros(size(b{l}));
        
        offset = offset+layers(l+1);
    end
    
    % Do forward pass to compute activations a and z
    [a, z] = model(X, W, b, L, f);
    
    % Apply backpropagation rule to compute gradient
    for i=1:N
        delta = dell(z{L}(:, i), Y(:, i));
        
        for l=L:-1:2
            grad_W{l} = grad_W{l} + delta * a{l-1}(:, i)';
            grad_b{l} = grad_b{l} + delta;
            delta = (W{l}'*delta).*df(z{l-1}(:, i));
        end
        grad_W{1} = grad_W{1} + delta*X(:, i)';
        grad_b{1} = grad_b{1} + delta;
    end
  
    % Stack all partial derivatives in a huge vector grad
    grad = zeros(0, 1);
    for l=1:L
        grad = cat(1, grad, grad_W{l}(:));
    end

    for l=1:L
        grad = cat(1, grad, grad_b{l});
    end
    
    % Evaluate loss
    obj = ell(z{L}, Y);
end

