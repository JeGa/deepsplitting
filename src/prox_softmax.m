function [x] = prox_softmax(u, c, rho)
%PROX_SOFTMAX Summary of this function goes here
%   Detailed explanation goes here

    n = size(u, 1);
    
    f = @(x) -log(exp(x(c)) / sum(exp(x), 1)) + (0.5 * rho) * sum((x-u).^2, 1);
    Id = eye(n);
    w = @(x) exp(x) ./ sum(exp(x), 1);
    %w = @(x) exp(x-max(x)) ./ sum(exp(x-max(x)), 1);
    
    grad = @(x) w(x) - Id(:, c) + rho * (x-u);
      
    
    M = @(x) (diag(1 ./ (w(x) + rho)));
    
    inv_Hess = @(x) M(x) * (Id + w(x)*w(x)' / (1 - w(x)'*M(x)*w(x)));
    %inv_Hess = @(x) Id;
    
    max_iter = 2500;
    tol = 1e-6;
    alpha = 0.01;
    beta = 0.95;
    
    %warmstart
    x = u;
    
    for i=1:max_iter
        g = grad(x);
        R = inv_Hess(x);
        p = -R*grad(x);
        lambda = g'*R*g;

        if(lambda/2 < tol)
            break;
        end
        
        t = 1;
        while(f(x + t*p) > f(x) + alpha*t*p'*g)
            t = beta * t;
        end

        x = x + t*p;
    end
    
    if(lambda/2 >= tol)
        warning('Did not reach tol %f within %d iterations.', tol, max_iter);
    end
end

