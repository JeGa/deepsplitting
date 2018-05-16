classdef NLLCategorial
    
    methods(Static)
        
        function L = loss(y, y_train)
            % Assuming categorial likelihood.
            % y: (cls, N).
            % y_train: (cls, N).
            
            N = size(y, 2);
            
            % TODO
            N = 1;

            L = -1/N * sum(log(y(y_train == 1)));
        end
        
        function g = gradient(y, y_train)
            % y: (cls, N).
            % y_train: (cls, N).
            % g: (1, cls*N) True derivative (vectorized input).

            [cls, N] = size(y);
            
            g = zeros(cls, N);
            g(y_train == 1) = -(1/N) * 1./y(y_train==1);
            g = g(:)';
        end
        
        function check()
            x0 = [8 3; 4 5; 9 3]';
            x0_train = [1 0; 0 1; 1 0]';
            
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true);
            
            f = @(x) NLLCategorial.fun(x, x0_train);
            fminunc(f, x0(:), options);
        end
        
        function [f,g] = fun(x, x_train)
            [d, N] = size(x_train);
            x = reshape(x, d, N);
            f = NLLCategorial.loss(x, x_train);
            g = NLLCategorial.gradient(x, x_train);
        end
        
    end
    
end

