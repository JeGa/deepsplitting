classdef LeastSquares
    
    methods(Static)
        
        function [L] = loss(y, y_train)
            % y: (cls, N).
            % y_train: (cls, N).
            
            N = size(y,2);
            L = (1/(2*N)) * norm(y - y_train)^2;
        end
        
        function [g] = gradient(y, y_train)
            % y: (cls, N).
            % y_train: (cls, N).
            
            N = size(y,2);

            g = (1/N) * (y - y_train);
        end
        
    end
    
end

