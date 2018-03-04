classdef Network
    properties
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
            [obj] = forwardpass(obj, X_train);
        end

        function obj = forwardpass(obj, X_train)
            L = size(obj.W, 2);
            n = size(X_train, 2);

            obj.z{1} = obj.W{1}*X_train + obj.b{1} * ones(1, n);

            for l=1:L-1
                % Apply activation function.
                obj.a{l} = obj.h(obj.z{l});

                % Apply linear mapping.
                obj.z{l+1} = obj.W{l+1}*obj.a{l} + obj.b{l+1} * ones(1, size(obj.a{l}, 2));
            end
        end
        
        function [obj, y] = fp(obj, X_train)
            % Returns the network output from the last layer.
            obj = obj.forwardpass(X_train);
            y = obj.z{size(obj.W, 2)};
        end
        
        function [obj] = train(obj, X_train, y_train, params)
            % params: Struct with
            %   stepsize, iterations, plot.
            
            for i=1:params.iterations
                [obj,y] = obj.fp(X_train);
                L = obj.loss.loss(y,y_train);
                
                disp(['Loss: ', num2str(L), ' (', num2str(i), ')']);
                
                if params.plot
                    obj.plot_result(X_train, y, 2);
                end
                    
                [obj, dW, db] = obj.gradient(X_train, y_train, y);
                
                for j=1:size(dW, 2)
                    obj.W{j} = obj.W{j} - params.stepsize * dW{j};
                    obj.b{j} = obj.b{j} - params.stepsize * db{j};
                end
            end
            
            [~,y] = obj.fp(X_train);
            L = obj.loss.loss(y,y_train);
                
            disp(['Loss: ', num2str(L)]);
        end
        
        function [obj, dW, db] = gradient(obj, X_train, y_train, y)
            layers = size(obj.W, 2);
            
            [g_loss] = obj.loss.gradient(y, y_train);
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
        
        function [dW] = Dw(~, error, a)
            a = a';
            
            out_dim = size(error, 2);
            in_dim = size(a, 2);
            
            error_shaped = repelem(error, 1, in_dim);
            a_shaped = repmat(a, 1, out_dim);
            
            dW = reshape(sum(error_shaped .* a_shaped,1), in_dim, out_dim)';
        end
        
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
