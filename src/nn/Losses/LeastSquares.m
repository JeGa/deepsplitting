classdef LeastSquares
    
    methods(Static)
        
        function [L] = loss(y, y_train)
            N = size(y,2);

            y_true = zeros(size(y));
            ind = sub2ind(size(y), y_train', 1:size(y_train',2));
            y_true(ind) = 1;

            L = (1/(2*N)) * norm(y - y_true)^2;
        end
        
        function [g] = gradient(y, y_train)
            N = size(y,2);

            y_true = zeros(size(y));
            ind = sub2ind(size(y), y_train', 1:size(y_train',2));
            y_true(ind) = 1;

            g = (1/(N)) * (y - y_true);
        end
        
    end
    
end

