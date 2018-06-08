classdef ProxPropNetwork < Network
    methods
        function obj = ProxPropNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);
        end
        
        function obj = train(obj, X_train, y_train, params)
            % params: Struct with
            %   iterations, tau (proximal operator factor).
            
            for i = 1:params.iterations
                % y is before parameter update.
                [obj, y] = obj.proxprop_step(params.tau, params.tau_theta, X_train, y_train);
                
                
                disp(['Loss = ', num2str(obj.loss.loss(y, y_train)), ' (', num2str(i), '/', num2str(params.iterations), ')']);
            end
        end
    end
    
    methods(Access=private)
        function [obj, y] = proxprop_step(obj, tau, tau_theta, X_train, y_train)
            % Forward-pass.
            [obj, y] = obj.fp(X_train);
            
            % With last linear layer.
            L = size(obj.layers, 2)-1;
            
            N = size(X_train, 2);
            
            % Last layer.
            obj.a{L-1} = obj.a{L-1} - tau * (obj.W{L}' * obj.loss.gradient(y, y_train));
            
            obj.W{L} = obj.W{L} - tau * obj.loss.gradient(y, y_train) * obj.a{L-1}';
            obj.b{L} = obj.b{L} - tau * sum(obj.loss.gradient(y, y_train), 2);
            
            % Other layers.
            for i = L-1:-1:1
                obj.z{i} = obj.z{i} - obj.dh(obj.z{i}) .* (obj.h(obj.z{i}) - obj.a{i});
                
                if i ~= 1
                    % Save before update.
                    a = obj.a{i-1};
                    
                    % Forward pass with new z.
                    yi = obj.W{i} * obj.a{i-1} + obj.b{i} * ones(1, N) - obj.z{i};
                    % Gradient step.
                    obj.a{i-1} = obj.a{i-1} - obj.W{i}' * yi;
                else
                    % a0 = X_train;
                    a = X_train;
                end
                
                % Prox operator.
                a_aug = [a;ones(1, N)];
                
                A = a_aug*a_aug' + (1/tau_theta)*eye(size(a_aug, 1));
                B = obj.z{i}*a_aug' + (1/tau_theta)*[obj.W{i},obj.b{i}];
                
                theta = (A' \ B')';
                
                obj.W{i} = theta(:,1:end-1);
                obj.b{i} = theta(:,end);
            end
        end
    end
end

