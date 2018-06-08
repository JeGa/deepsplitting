classdef Network
    properties(Access=protected)
        W % Weights
        b % Biases
        z % Variables for linearities.
        a % Variables for activations.
        h % Activation function.
        dh % Derivative of activation function.
        loss % Object for calculating loss and its gradient.
        layers % Array with number of hidden units (including input and output).
    end
    
    methods
        function obj = Network(layers, h, dh, loss, X_train)
            % Number of layers (including last linear layer).
            L = size(layers, 2)-1;
            
            obj.layers = layers;
            
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
            y = obj.z{end};
        end
        
        function [obj, L, y] = f(obj, X_train, y_train)
            % Returns the network loss.
            [obj, y] = obj.fp(X_train);
            L = obj.loss.loss(y, y_train);
        end
        
        function obj = check_gradients(obj, layers, X_train, y_train)
            % Checks the gradients computed by backpropagation.
            
            x0 = obj.to_vec(obj.W, obj.b, 1);
                       
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true, 'Display', 'iter');
            
            f = @(x) obj.fun(x, layers, X_train, y_train);
            [x, ~] = fminunc(f, x0, options);
            
            [W_min, b_min] = obj.to_mat(x, layers, 1);
            
            %obj.W = W_min;
            %obj.b = b_min;
        end
        
        function obj = check_jacobian(obj, layers, X_train, y_train)
           % Checks the jacobian of the feed forward part.
           
           x0 = obj.to_vec(obj.W, obj.b, 1);
           
           options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
               'SpecifyObjectiveGradient', true, 'CheckGradients', true, 'Display', 'iter');
           
           f = @(x) obj.fun_jacobian(x, layers, X_train, y_train);
           [x, ~] = fminunc(f, x0, options);
           
           [W_min, b_min] = obj.to_mat(x, layers, 1);
           
           %obj.W = W_min;
           %obj.b = b_min;
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
            
            num_layers = size(W, 2);
            
            g_loss = loss.gradient(y, y_train);
            dL = g_loss'; % (samples,dim).
            
            error = cell(1, num_layers);
            dW = cell(1, num_layers);
            db = cell(1, num_layers);
            
            % Last layer.
            error{num_layers} = dL;
            db{num_layers} = sum(dL, 1)';
            dW{num_layers} = obj.Dw(dL, obj.a{num_layers-1});
            
            % Other layers.
            for i = num_layers-1:-1:2
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
        
        function [obj, dW, db, y] = jacobian_eval_noloss(obj, W, b, X_train)
            % Compute the jacobian of f(({W_j},{b_j})). Note that there is no
            % error function. The errors are now Jacobians. The W matrices are
            % vectorized in row-major order.
            obj = obj.forwardpass(W, b, X_train);
            y = obj.z{end};
            
            num_layers = size(W, 2);
            N = size(X_train, 2);
            
            error = cell(1, num_layers);
            dW = cell(1, num_layers); 
            db = cell(1, num_layers);
            
            % Last layer: There is no error.
            error{num_layers} = eye(numel(obj.z{num_layers}));
            dW{num_layers} = obj.dzdw(obj.a{num_layers-1}, size(obj.z{num_layers}, 1));
            
            ddim = size(obj.z{num_layers}, 1);
            db{num_layers} = repmat(eye(ddim), N, 1);
            
            flatten = @(x) x(:);
            
            % Other layers.
            for i = num_layers-1:-1:2
                k = repmat(obj.W{i+1}, 1, N);
                d = repmat(obj.dh(flatten(obj.z{i}))', size(obj.W{i+1}, 1), 1);
                A = k .* d; % (c, d*N).
                
                [ys, xs] = size(obj.W{i+1});
                iA = flatten(repmat(reshape(1:ys*N, ys, N), xs, 1));
                jA = repelem(1:xs*N, ys)';
                Ad = sparse(iA, jA, A(:));
                error{i} = error{i+1} * Ad;
                
                [ddim, ~] = size(obj.W{i});
                
                dW{i} = error{i} * obj.dzdw(obj.a{i-1}, ddim);
                db{i} = error{i} * repmat(eye(ddim), N, 1);
            end
            
            % First layer.
            k = kron(eye(N), obj.W{1+1});
            d = obj.dh(flatten(obj.z{1}))';
            error{1} = error{1+1} * (k .* d);

            [ddim, ~] = size(obj.W{1});

            dW{1} = error{1} * obj.dzdw(X_train, ddim);
            db{1} = error{1} * repmat(eye(ddim), N, 1);
        end
        
        function D = dzdw(~, A, d)
            % A: (vdim, N), d: In R.
            % z(vec_row_major(W)) = vec_column_major(W*A) with W*a of shape (d, N).
            % W is row major because its easier to vectorize that way.
            % Returns dz/dw jacobian matrix.
            [vdim, N] = size(A);
            
            % Vector diagonal matrix.
            % 1..1 0..0 0..0
            % 0..0 1..1 0..0
            % 0..0 0..0 1..1
            % 1..1 0..0 0..0
            % ...
            D = kron(repmat(eye(d), N, 1), ones(1, vdim));
            A = repmat(repelem(A', d, 1), 1, d);
            
            D = D .* A;
        end
        
        function [obj, J, y] = jacobian_noloss_matrix(obj, X_train)
            % Jc(x) (Jacobian with row-major vectorization) and c(x).
            [obj, dW, db, y] = obj.jacobian_eval_noloss(obj.W, obj.b, X_train);
            
            J = obj.to_jacobian(dW, db);
        end
        
        function [L, g] = fun(obj, x, layers, X_train, y_train)
            [Wc, bc] = obj.to_mat(x, layers, 1);
            
            [~, dW, db, L, ~] = obj.gradient_eval(Wc, bc, X_train, y_train, obj.loss);
            g = obj.to_vec(dW, db, 1);
        end
        
        function [L, g] = fun_jacobian(obj, x, layers, X_train, y_train)
            [Wc, bc] = obj.to_mat(x, layers, 1);
            
            % Row major order.
            [~, dW, db, y] = obj.jacobian_eval_noloss(Wc, bc, X_train);
                       
            J = obj.to_jacobian(dW, db);
            
            gradient_loss = obj.loss.gradient(y, y_train);
            
            g = J'*gradient_loss(:);
            
            L = obj.loss.loss(y, y_train);
        end
        
        function x = to_vec(~, W, b, order)
            % order: 1 = row-major, 2 = column-major.
            flatten = @(x) x(:);
            
            if order == 1
                flatten_tp = @(x) flatten(x');
                
                W_flat = cellfun(flatten_tp, W, 'un', 0);
                b_flat = cellfun(flatten_tp, b, 'un', 0);
            elseif order == 2
                W_flat = cellfun(flatten, W, 'un', 0);
                b_flat = cellfun(flatten, b, 'un', 0);
            end

            W_vec = cat(1, W_flat{:});
            b_vec = cat(1, b_flat{:});
            
            x = [W_vec; b_vec];
        end
            
        function [W, b] = to_mat(~, x, layers, order)
            % order: 1 = row-major, 2 = column-major.
            L = size(layers, 2)-1;
            
            W = cell(1, L);
            b = cell(1, L);
            
            W_size = layers(1:end-1) * layers(2:end)';
            
            W_vec = x(1:W_size, 1);
            b_vec = x(W_size+1:end, 1);
            
            from_W = 1;
            from_b = 1;
            
            for i=1:L
               to_W = from_W + layers(i+1) * layers(i) - 1;
               to_b = from_b + layers(i+1) - 1;
                
               if order == 1
                   W{i} = reshape(W_vec(from_W:to_W), layers(i), layers(i+1))';
               elseif order == 2
                   W{i} = reshape(W_vec(from_W:to_W), layers(i+1), layers(i));
               end
               
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
        
        function J = to_jacobian(~, dW, db)
            % Stacks the individual Jacobians computed by backpropagation
            % to one big Jacobian. (Usually its possible to directly work
            % with the individual J_W1, J_W2, ... and the stacking is not
            % required.
            JW = [dW{:}];
            Jb = [db{:}];
            J = [JW, Jb];
        end 
    end
end
