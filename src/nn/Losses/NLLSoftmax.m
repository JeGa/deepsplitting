classdef NLLSoftmax
   % Combines Softmax and NLLCategorial.
   
    methods
        
        function L = loss(~, y, y_train)
            y = Softmax.softmax(y);
            
            L = NLLCategorial.loss(y, y_train);
        end
        
        function g = gradient(~, y, y_train)
            % g: (d, N) Matrix form gradient.
            [d, N] = size(y);
            
            y_softmax = Softmax.softmax(y);
            
            g_loss = NLLCategorial.gradient(y_softmax, y_train);
            g_softmax = Softmax.gradient(y);
            
            g = g_loss * g_softmax;
            g = reshape(g, d, N);
        end
        
        function check(~)
            x0 = [2, 5; 1 2]';
            x0_train = [1, 0; 0 1]';
            
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true);
            
            f = @(x) NLLSoftmax.fun(x, x0_train);
            fminunc(f, x0(:), options);
        end
        
        function [f,g] = fun(~, x, x_train)
            [d, N] = size(x_train);
            x = reshape(x, d, N);
            
            f = NLLSoftmax.loss(x, x_train);
            g = NLLSoftmax.gradient(x, x_train);
            g = g(:);
        end
        
        function z = primal_update(~, y, lambda, y_train, rho)
            % Augmented lagrangian primal update.
            z = zeros(size(y));
            
            r = y + lambda/rho;
            
            for i = 1:size(y_train, 2)
                [~, cls] = max(y_train(:,i));
                z(:, i) = prox_cross_entropy(r(:, i), 1/rho, cls);
            end
        end
        
    end
    
end