classdef Softmax

    methods(Static)
        
        function y = softmax(x)
            % x: (d, N).
            y = exp(x) ./ repmat(sum(exp(x), 1), size(x, 1), 1);
        end
        
        function g = gradient(x)
            % x: (d, N).
            % g: ((d*N),(d*N)) Jacobian.

            % (d*N).
            s = softmax(x(:));
            % ((d*N),(d*N)).
            g = diag(s) - (s * s');

            % The Jacobian is a block diagonal matrix.
            % Set the non-related parts to zero.
            [d, N] = size(x);
            Q = kron(eye(N), ones(d));

            g = Q .* g;
        end
        
    end
    
end

