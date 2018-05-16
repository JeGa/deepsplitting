classdef Softmax

    methods(Static)
        
        function y = softmax(x)
            % x: (d, N).
            y = exp(x) ./ sum(exp(x), 1);
        end
        
        function g = gradient(y)
            % y: (d, N).
            % g: ((d*N),(d*N)) (True Jacobian (vectorized input)).

            % (d*N).
            s = Softmax.softmax(y);
            s = s(:);
            
            % ((d*N),(d*N)).
            g = diag(s) - (s * s');

            % The Jacobian is a block diagonal matrix.
            % Set the non-related parts to zero.
            [d, N] = size(y);
            Q = kron(eye(N), ones(d));

            g = Q .* g;
        end
        
    end
    
end

