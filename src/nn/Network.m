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
                obj.W{l} = 0.1*randn(layers(l+1), layers(l));
                obj.b{l} = 0.1*randn(layers(l+1), 1);
                
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
            %   stepsize, iterations, plot.
            
            for i=1:params.iterations
                [obj, L, y] = obj.f(X_train, y_train);
                
                if params.plot
                    obj.plot_result(X_train, y, 2);
                end
                    
                [obj, dW, db] = obj.gradient(X_train, y_train, y);
                
                % Step directions.
                sW = cellfun(@(x) -x, dW, 'UniformOutput', 0);
                sb = cellfun(@(x) -x, db, 'UniformOutput', 0);
                
                if params.linesearch == 0
                    stepsize = params.stepsize;
                elseif params.linesearch == 1
                    stepsize = obj.armijo(obj.W, obj.b, sW, sb, X_train, y_train, params.beta, params.gamma);
                elseif params.linesearch == 2
                    stepsize = obj.pw();
                end
                
                disp(['Loss: ', num2str(L), ', stepsize: ', num2str(stepsize), ' (', num2str(i), ')']);
                
                [obj.W, obj.b] = obj.update_weights(stepsize, obj.W, obj.b, sW, sb);
            end
            
            [obj, L, ~] = obj.f(X_train, y_train);
            disp(['Loss: ', num2str(L)]);
        end
        
        function [obj, L, y] = f(obj, X_train, y_train)
            [obj,y] = obj.fp(X_train);
            L = obj.loss.loss(y, y_train);
        end
        
        % TODO
        function [] = check_gradients(obj)
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
        
        function [obj, dW, db] = gradient(obj, X_train, y_train, y)
            layers = size(obj.W, 2);
            
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
                error{i} = error{i+1} * obj.W{i+1} .* obj.dh(obj.z{i})';
                db{i} = sum(error{i}, 1)';
                dW{i} = obj.Dw(error{i}, obj.a{i-1});
            end
            
            % First layer.
            error{1} = error{1+1} * obj.W{1+1} .* obj.dh(obj.z{1})';
            db{1} = sum(error{1}, 1)';
            dW{1} = obj.Dw(error{1}, X_train);
        end
        
        function dW = Dw(~, error, a)
            a = a';
            
            out_dim = size(error, 2);
            in_dim = size(a, 2);
            
            error_shaped = repelem(error, 1, in_dim);
            a_shaped = repmat(a, 1, out_dim);
            
            dW = reshape(sum(error_shaped .* a_shaped,1), in_dim, out_dim)';
        end
        
        function sigma = armijo(obj, W, b, sW, sb, X_train, y_train, beta, gamma)
            k = 1;
            
            while 1
                sigma = (beta^k) / beta;
                
                % x_new = x + sigma * s
                [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
                
                % f(x_new) - f(x)
                [~, L_new, ~] = obj.f_eval(W_new, b_new, X_train, y_train);
                [~, L, ~] = obj.f_eval(W, b, X_train, y_train);
                f_new =  L_new - L;
                
                flatten = @(x) x(:);
                W_sum = sum(cellfun(@(W, sW) sum(flatten(W .* sW)), W, sW));
                b_sum = sum(cellfun(@(b, sb) sum(flatten(b .* sb)), b, sb));
                
                if f_new <= sigma * gamma * (W_sum + b_sum)
                   break;
                end

                k = k + 1;
            end
        end
        
        % TODO
        function [sigma] = pw(obj)
        end
        
        % TODO: Remove.
        function plot_result(~, X_train, y, f)
            [~,C] = max(y, [], 1);
            C = C - 1;

            % y is of shape (cls, samples).
            figure(f);
            scatter(X_train(1,:), X_train(2,:), 10, C);
            axis equal;
            drawnow
        end
            
    end
end
