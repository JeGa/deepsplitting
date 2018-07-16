function [J, v] = backprop_jacobian(x, layers, f, df, X, Y)
    % Number of layers (including last linear layer).
    N = size(X, 2);
    c = layers(end);
    
    L = size(layers, 2)-1;
    
    % Weights
    W = cell(1, L);
    % Biases
    b = cell(1, L);
    
    J_W = cell(1, L);
    J_b = cell(1, L);
    
    % Read off network weights.
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
    
    % Do forward pass to compute activations a and z
    [a, z] = model(X, W, b, f);
    
    % Apply backpropagation rule to compute Jacobian
    
    for i=1:N
       for k=1:c
           Id = eye(c);
           J_W{L} = cat(1, J_W{L}, reshape(kron(Id(k, :), a{L-1}(:,i)'), layers(L)*layers(L+1), 1)'); 
           J_b{L} = cat(1, J_b{L}, Id(k, :));            
           delta = diag(df(z{L-1}(:,i)))*W{L}(k, :)';
           for l = L-1:-1:2
               J_W{l} = cat(1, J_W{l}, reshape(delta*a{l-1}(:,i)', layers(l)*layers(l+1), 1)'); 
               J_b{l} = cat(1, J_b{l}, delta');            
               delta = diag(df(z{l-1}(:,i)))*W{l}'*delta;
           end
           J_W{1} = cat(1,J_W{1}, reshape(delta*X(:,i)', layers(1)*layers(2), 1)');
           J_b{1} = cat(1,J_b{1}, delta');
       end
    end
  
    J =[];
    for l = 1:L
       J = cat(2,J,J_W{l});
    end
    for l = 1:L
        J = cat(2,J,J_b{l});
    end
    
    v = z{L}-Y;
end

