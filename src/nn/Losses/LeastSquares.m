classdef LeastSquares
    
    properties(Access=public)
        C
    end
    
    methods
        
        function obj = LeastSquares(C)
            obj.C = C;
        end
        
        function L = loss(obj, y, y_train)
            % y: (cls, N).
            % y_train: (cls, N).
            L = obj.C * 0.5 * sum(sum((y - y_train).^2));
        end
        
        function g = gradient(obj, y, y_train)
            % y: (cls, N).
            % y_train: (cls, N).
            g = obj.C * (y - y_train);
        end
        
        function z = primal_update(obj, y, lambda, y_train, rho)
            % Augmented lagrangian primal update.
            z = (obj.C * y_train + rho * y + lambda) / (obj.C + rho);
        end
        
    end
    
end

