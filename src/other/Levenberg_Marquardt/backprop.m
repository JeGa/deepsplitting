function [obj, J, v] = backprop(x, layers, f, df, ell, dell, X, Y)
    % Number of layers (including last linear layer).
    N = size(X, 2);

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
    [a, z] = model(X, W, b, L, f);
    
    % Apply backpropagation rule to compute Jacobian
    
    for i=1:N
       delta = eye(size(Y,1));
       for l = L:-1:2
           J_W{l} = cat(1, J_W{l}, reshape(delta*a{l-1}(:,i)', layers(l)*layers(l+1), size(Y,1))'); 
           J_b{l} = cat(1, J_b{l}, delta');              
           delta = diag(df(z{l-1}(:,i)))*W{l}'*delta;
       end
       J_W{1} = cat(1,J_W{1}, reshape(delta*X(:,i)', layers(1)*layers(2), size(Y,1))');
       J_b{1} = cat(1,J_b{1}, delta');
       
    end
  
    J =[];
    for l = 1:L
        J = cat(2,J,J_W{l});
    end
    for l = 1:L
        J = cat(2,J,J_b{l});
    end
    obj = ell(z{L},Y);
    v = dell(z{L},Y)';
    
end

