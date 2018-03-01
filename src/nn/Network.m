classdef Network
    properties
        W % Weights
        b % Biases
        z % Variables for linearities.
        a % Variables for activations.
        lambda % Lagrange multipliers.
        h % Activation function.
        loss % Object for calculating loss and its gradient.
    end
    
    methods
        function obj = Network(layers, h, loss, X_train)
            % Number of layers (including last linear layer).
            L = size(layers, 2)-1;

            obj.W = cell(1, L);
            obj.b = cell(1, L);
            obj.z = cell(1, L);
            obj.a = cell(1, L-1);
            obj.lambda = cell(1, L);
            
            obj.h = h;
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
        
        function [obj] = train(obj, X_train, y_train, stepsize)

            for i=1:10
                [dW, db] = obj.gradient(X_train, y_train);
                
                for j=1:size(layers)
                    obj.W{j} = obj.W{j} - stepsize * dW{j};
                    obj.b{j} = obj.b{j} - stepsize * db{j};
                end
                
                [~,y] = obj.fp(X_train);
                L = obj.loss.loss(y,y_train);
                
                disp(['Loss: ', num2str(L)]);
            end
        end
        
        function [obj] = gradient(obj, X_train, y_train)
            
        end
        
    end
end
