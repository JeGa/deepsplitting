classdef Softmax

    methods
        
        function [ y ] = softmax(a)
            y = exp(a) ./ sum(exp(a), 1);
        end
        
        function [g] = gradient(y)
            % y: CxN, g: (C*N)x(C*N).

            % C*N.
            s = softmax(y(:));
            g = diag(s) - (s * s');

            % The Jacobian is a block diagonal matrix.
            % Set the non-related parts to zero.
            [C, N] = size(y);
            Q = kron(eye(N), ones(C));

            g = Q .* g;
        end
        
    end
    
end

