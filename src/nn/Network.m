classdef Network
    properties(Access=protected)
        W % Weights
        b % Biases
        z % Variables for linearities.
        a % Variables for activations.
        h % Activation function.
        dh % Derivative of activation function.
        loss % Object for calculating loss and its gradient.
    end
    
    methods
        
        function obj = Network(layers, h, dh, loss, X_train)
            % Number of layers (including last linear layer).
            L = size(layers, 2)-1;
            
            obj.W = cell(1, L);
            obj.b = cell(1, L);
            obj.z = cell(1, L);
            obj.a = cell(1, L-1);
            
            obj.h = h;
            obj.dh = dh;
            obj.loss = loss;
            
            % Initialize network weights.
            for l=1:L
                obj.W{l} = 0.1 * randn(layers(l+1), layers(l));
                obj.b{l} = 0.1 * randn(layers(l+1), 1);
            end
            
            % Perform a forward pass to find a feasible initialization of all variables.
            obj = forwardpass(obj, obj.W, obj.b, X_train);
        end
        
        function [obj, y] = fp(obj, X_train)
            % Returns the network output from the last layer.
            obj = obj.forwardpass(obj.W, obj.b, X_train);
            y = obj.z{size(obj.W, 2)};
        end
        
        function [obj, L, y] = f(obj, X_train, y_train)
            % Returns the network loss.
            [obj, y] = obj.fp(X_train);
            L = obj.loss.loss(y, y_train);
        end
        
        function obj = check_gradients(obj, layers, X_train, y_train)
            % Checks the gradients computed by backpropagation.
            [x0, W_size, ~] = obj.to_vec(obj.W, obj.b);
                       
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true);
            
            f = @(x) obj.fun(x, W_size, layers, X_train, y_train);
            [x, ~] = fminunc(f, x0, options);
            
            [W_min, b_min] = obj.to_mat(x, W_size, layers);
            
            obj.W = W_min;
            obj.b = b_min;
        end
    end
    
    methods(Abstract=true)
        obj = train(obj, X_train, y_train, params);
    end
    
    methods(Access=protected)
        function [obj, L, y] = f_eval(obj, W, b, X_train, y_train, loss)
            % Evaluate at some specific point.
            obj = obj.forwardpass(W, b, X_train);
            y = obj.z{size(W, 2)};
            L = loss.loss(y, y_train);
        end
        
        function [obj, y] = fp_eval(obj, W, b, X_train)
            obj = obj.forwardpass(W, b, X_train);
            y = obj.z{size(W, 2)};
        end
        
        function obj = forwardpass(obj, W, b, X_train)
            L = size(W, 2);
            n = size(X_train, 2);
            
            obj.z{1} = W{1}*X_train + b{1} * ones(1, n);
            
            for l=1:L-1
                % Apply activation function.
                obj.a{l} = obj.h(obj.z{l});
                
                % Apply linear mapping.
                obj.z{l+1} = W{l+1}*obj.a{l} + b{l+1} * ones(1, size(obj.a{l}, 2));
            end
        end
        
        function [W, b] = update_weights(~, stepsize, W, b, sW, sb)
            step = @(A, dA) A + stepsize * dA;
            W = cellfun(step, W, sW, 'un', 0);
            b = cellfun(step, b, sb, 'un', 0);
        end
        
        function [obj, dW, db, L, y] = gradient_eval(obj, W, b, X_train, y_train, loss)
            % Gradient of L(f({W_j},{b_j})) at (W, b).
            [obj, L, y] = obj.f_eval(W, b, X_train, y_train, loss);
            
            layers = size(W, 2);
            
            g_loss = loss.gradient(y, y_train);
            dL = g_loss'; % (samples,dim).
            
            error = cell(1, layers);
            dW = cell(1, layers);
            db = cell(1, layers);
            
            % Last layer.
            error{layers} = dL;
            db{layers} = sum(dL, 1)';
            dW{layers} = obj.Dw(dL, obj.a{layers-1});
            
            % Other layers.
            for i = layers-1:-1:2
                error{i} = error{i+1} * W{i+1} .* obj.dh(obj.z{i})';
                db{i} = sum(error{i}, 1)';
                dW{i} = obj.Dw(error{i}, obj.a{i-1});
            end
            
            % First layer.
            error{1} = error{1+1} * W{1+1} .* obj.dh(obj.z{1})';
            db{1} = sum(error{1}, 1)';
            dW{1} = obj.Dw(error{1}, X_train);
        end
        
        function [obj, dW, db, L, y] = gradient(obj, X_train, y_train)
            [obj, dW, db, L, y] = obj.gradient_eval(obj.W, obj.b, X_train, y_train, obj.loss);
        end
        
        function dW = Dw(~, error, a)
            a = a';
            
            out_dim = size(error, 2);
            in_dim = size(a, 2);
            
            error_shaped = repelem(error, 1, in_dim);
            a_shaped = repmat(a, 1, out_dim);
            
            dW = reshape(sum(error_shaped .* a_shaped,1), in_dim, out_dim)';
        end
        
        function [obj, dW, db, y] = gradient_eval_noloss(obj, W, b, X_train)
            % Compute gradient of f(({W_j},{b_j})). Note that there is no
            % error function. The errors are now Jacobians.
            obj = obj.forwardpass(W, b, X_train);
            y = obj.z{size(W, 2)};
            
            layers = size(W, 2);
            N = size(X_train, 2);
            
            error = cell(1, layers);
            dW = cell(1, layers); 
            db = cell(1, layers);
            
            % Last layer: There is no error.
            error{layers} = eye(numel(obj.z{layers}));
            dW{layers} = obj.dzdw(obj.a{layers-1}, size(obj.z{layers}, 1));
            
            ddim = size(obj.z{layers}, 1);
            db{layers} = repmat(eye(ddim), N, 1);
            
            flatten = @(x) x(:);
            
            % Other layers.
            for i = layers-1:-1:2
                k = kron(eye(N), obj.W{i+1});
                d = obj.dh(flatten(obj.z{i}))';
                error{i} = error{i+1} * (k .* d);
                
                [ddim, ~] = size(obj.W{i});
                
                dW{i} = error{i} * obj.dzdw(obj.a{i-1}, ddim);
                db{i} = error{i} * repmat(eye(ddim), N, 1); % TODO: Faster?
            end
            
            % First layer.
            k = kron(eye(N), obj.W{1+1});
            d = obj.dh(flatten(obj.z{1}))';
            error{1} = error{1+1} * (k .* d);

            [ddim, ~] = size(obj.W{1});

            dW{1} = error{1} * obj.dzdw(X_train, ddim);
            db{1} = error{1} * repmat(eye(ddim), N, 1); % TODO: Faster?
        end
        
        function D = dzdw(~, A, d)
            % Returns the dz/dw matrix in row major order.
            [vdim, N] = size(A);
            
            D = kron(repmat(eye(d), N, 1), ones(1, vdim));
            A = repmat(repelem(A', d, 1), 1, d);
            
            D = D .* A;
        end
        
        function [L, g] = fun(obj, x, W_size, layers, X_train, y_train)
            [Wc, bc] = obj.to_mat(x, W_size, layers);
            
            [~, dW, db, L, ~] = obj.gradient_eval(Wc, bc, X_train, y_train, obj.loss);
            g = obj.to_vec(dW, db);
        end
        
        function [x, W_size, b_size] = to_vec(~, W, b)
            % Column major order for W.
            flatten = @(x) x(:);
            
            W_flat = cellfun(flatten, W, 'un', 0);
            b_flat = cellfun(flatten, b, 'un', 0);
            
            W_vec = cat(1, W_flat{:});
            b_vec = cat(1, b_flat{:});
            
            x = [W_vec; b_vec];
            
            W_size = size(W_vec);
            b_size = size(b_vec);
        end
            
        function [W, b] = to_mat(~, x, W_size, layers)
            L = size(layers, 2)-1;
            
            W = cell(1, L);
            b = cell(1, L);
            
            W_vec = x(1:W_size(1), 1);
            b_vec = x(W_size(1)+1:end, 1);
            
            from_W = 1;
            from_b = 1;
            
            for i=1:L
               to_W = from_W + layers(i+1) * layers(i) - 1;
               to_b = from_b + layers(i+1) - 1;
                
               W{i} = reshape(W_vec(from_W:to_W), layers(i+1), layers(i));
               b{i} = reshape(b_vec(from_b:to_b), layers(i+1), 1);
               
               from_W = to_W + 1;
               from_b = to_b + 1;
            end
        end
        
        function d = directional_derivative(~, dW, db, sW, sb)
            % Directional derivative at W,b in direction s.
            % vec(dW, db)' * vec(sW, sb) = grad(x)'*s.
            flatten = @(x) x(:);
            W_sum = sum(cellfun(@(dW, sW) sum(flatten(dW .* sW)), dW, sW));
            b_sum = sum(cellfun(@(db, sb) sum(flatten(db .* sb)), db, sb));
            d = W_sum + b_sum;
        end
        
    end
end
