classdef CrossEntropy
    
    methods(Static)
        
        function [L] = loss(y, y_train)
            % Cross-entropy loss (ML/-log on softmax).
            N = size(y,2);
            
            ind = sub2ind(size(y), y_train', 1:size(y_train',2));

            L = -1/N * sum(log(y(ind)));
        end
        
        function [g] = gradient(y, y_train)
            % Returns gradient in column-major order.
            % y: CxN, y_train: Nx1, g: 1xC*N.

            N = size(y,2);
            
            ind = sub2ind(size(y), y_train', 1:size(y_train',2));

            g = zeros(1, size(y,1) * N);
            g(ind) = -(1/N) *  1./y(ind);
        end
        
    end
    
end

