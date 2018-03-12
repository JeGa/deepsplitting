classdef Network
    properties(Access=private)
        W % Weights
        b % Biases
        z % Variables for linearities.
        a % Variables for activations.
        lambda % Lagrange multipliers.
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
            obj.lambda = cell(1, L);
            
            obj.h = h;
            obj.dh = dh;
            obj.loss = loss;
            
            % Initialize network weights.
            for l=1:L
                obj.W{l} = 0.1 * randn(layers(l+1), layers(l));
                obj.b{l} = 0.1 * randn(layers(l+1), 1);
                
                % Initialize Lagrange multipliers for equality constraints.
                obj.lambda{l} = ones(layers(l+1), size(X_train, 2));
            end
            
            % Perform a forward pass to find a feasible initialization of all variables.
            obj = forwardpass(obj, obj.W, obj.b, X_train);
        end
        
        function [obj, y] = fp(obj, X_train)
            % Returns the network output from the last layer.
            obj = obj.forwardpass(obj.W, obj.b, X_train);
            y = obj.z{size(obj.W, 2)};
        end
        
        function obj = train(obj, X_train, y_train, params)
            % params: Struct with
            %   linesearch parameters, iterations, plot.
            
            for i=1:params.iterations
                [obj, dW, db, L, y] = obj.gradient(X_train, y_train);
                
                % Step directions.
                sW = cellfun(@(x) -x, dW, 'UniformOutput', 0);
                sb = cellfun(@(x) -x, db, 'UniformOutput', 0);
                
                if params.linesearch == 1
                    stepsize = params.stepsize;
                elseif params.linesearch == 2
                    stepsize = obj.armijo(obj.W, obj.b, dW, db, ...
                        sW, sb, X_train, y_train, params.beta, params.gamma);
                elseif params.linesearch == 3
                    stepsize = obj.pw(obj.W, obj.b, dW, db, ...
                        sW, sb, X_train, y_train, params.gamma, params.eta, params.beta);
                end
                
                if mod(i, 50) == 0
                    disp(['Loss: ', num2str(L), ', stepsize: ', num2str(stepsize), ...
                        ' gradnorm: ', num2str(norm(obj.to_vec(dW, db))) ,' (', num2str(i), ')']);
                end
                
                if params.plot
                    obj.plot_result(X_train, y, 2);
                end
                
                [obj.W, obj.b] = obj.update_weights(stepsize, obj.W, obj.b, sW, sb);
            end
            
            [obj, L, ~] = obj.f(X_train, y_train);
            disp(['Loss: ', num2str(L)]);
        end
        
        function [obj, L, y] = f(obj, X_train, y_train)
            [obj, y] = obj.fp(X_train);
            L = obj.loss.loss(y, y_train);
        end
        
        function obj = check_gradients(obj, layers, X_train, y_train)
            [x0, W_size, ~] = obj.to_vec(obj.W, obj.b);
                       
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true);
            
            f = @(x) obj.fun(x, W_size, layers, X_train, y_train);
            [x, ~] = fminunc(f, x0, options);
            
            [W_min, b_min] = obj.to_mat(x, W_size, layers);
            
            obj.W = W_min;
            obj.b = b_min;
        end

        function [L, g] = fun(obj, x, W_size, layers, X_train, y_train)
            [Wc, bc] = obj.to_mat(x, W_size, layers);
            
            [~, dW, db, L, ~] = obj.gradient_eval(Wc, bc, X_train, y_train);
            g = obj.to_vec(dW, db);
        end
        
        function [x, W_size, b_size] = to_vec(~, W, b)
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
        
    end
    
    methods(Access=private)
        
        function [obj, L, y] = f_eval(obj, W, b, X_train, y_train)
            % Evaluate at some specific point.
            obj = obj.forwardpass(W, b, X_train);
            y = obj.z{size(W, 2)};
            L = obj.loss.loss(y, y_train);
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
            for j=1:size(sW, 2)
                W{j} = W{j} + stepsize * sW{j};
                b{j} = b{j} + stepsize * sb{j};
            end
        end
        
        function [obj, dW, db, L, y] = gradient_eval(obj, W, b, X_train, y_train)
            [obj, L, y] = obj.f_eval(W, b, X_train, y_train);
            
            layers = size(W, 2);
            
            g_loss = obj.loss.gradient(y, y_train);
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
            [obj, dW, db, L, y] = obj.gradient_eval(obj.W, obj.b, X_train, y_train);
        end
        
        function dW = Dw(~, error, a)
            a = a';
            
            out_dim = size(error, 2);
            in_dim = size(a, 2);
            
            error_shaped = repelem(error, 1, in_dim);
            a_shaped = repmat(a, 1, out_dim);
            
            dW = reshape(sum(error_shaped .* a_shaped,1), in_dim, out_dim)';
        end
        
        function sigma = armijo(obj, W, b, dW, db, sW, sb, X_train, y_train, beta, gamma)
            k = 1;
            
            while 1
                sigma = (beta^k) / beta;
                
                if obj.check_armijo(sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    break;
                end
                
                k = k + 1;
            end
        end
        
        function r = check_armijo(obj, sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
            % Returns true if armijo condition is satisfied.
            
            % x_new = x + sigma * s
            [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
            
            % f(x_new) - f(x)
            [~, L_new, ~] = obj.f_eval(W_new, b_new, X_train, y_train);
            [~, L_current, ~] = obj.f_eval(W, b, X_train, y_train);
            f_new = L_new - L_current;
            
            slope = obj.directional_derivative(dW, db, sW, sb);
            
            if f_new <= sigma * gamma * slope
                r = true;
            else
                r = false;
            end
        end
        
        function d = directional_derivative(~, dW, db, sW, sb)
            flatten = @(x) x(:);
            W_sum = sum(cellfun(@(dW, sW) sum(flatten(dW .* sW)), dW, sW));
            b_sum = sum(cellfun(@(db, sb) sum(flatten(db .* sb)), db, sb));
            d = W_sum + b_sum;
        end
        
        function sigma = pw(obj, W, b, dW, db, sW, sb, X_train, y_train, gamma, eta, beta)
            min_slope = eta * obj.directional_derivative(dW, db, sW, sb);
            
            sigma = 1;
            
            if obj.check_armijo(sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                if obj.check_pw(sigma, W, b, sW, sb, X_train, y_train, min_slope)
                    return;
                else
                    sigma_pos = beta;
                    k = 1;
                    
                    while obj.check_armijo(sigma_pos, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                        k = k + 1;
                        sigma_pos = beta^k;
                    end
                    
                    sigma_neg = beta^(-1) * sigma_pos;
                end
            else
                sigma_neg = beta^(-1);
                k = 1;
                
                while ~obj.check_armijo(sigma_neg, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    k = k + 1;
                    sigma_neg = beta^(-k);
                end
                
                sigma_pos = beta * sigma_neg;
            end
            
            while ~obj.check_pw(sigma_neg, W, b, sW, sb, X_train, y_train, min_slope)
                s = (sigma_neg + sigma_pos) * 0.5;
                if obj.check_armijo(s, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    sigma_neg = s;
                else
                    sigma_pos = s;
                end
            end
            
            sigma = sigma_neg;
        end
        
        function r = check_pw(obj, sigma, W, b, sW, sb, X_train, y_train, min_slope)
            % Returns true if second Powell-Wolfe condition is satisfied.
            
            % x_new = x + sigma * s
            [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
            
            % Slope at new point.
            [~, dW_new, db_new, ~, ~] = obj.gradient_eval(W_new, b_new, X_train, y_train);
            slope = obj.directional_derivative(dW_new, db_new, sW, sb);
            
            if slope >= min_slope
                r = true;
            else
                r = false;
            end
        end
        
    end
end
