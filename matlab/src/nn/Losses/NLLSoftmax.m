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
        
        function [f, g] = fun(~, x, x_train)
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
        
        function d_min = minimize_linearized_penalty(obj, J, y, y_train, mu)
            options = optimoptions(@fminunc, 'SpecifyObjectiveGradient', true, 'Display', 'off');
            
            d0 = ones(size(J,2), 1);
            
            f = @(d) obj.fun_linearized_penalty(d, J, y, y_train, mu);
            [d_min, ~] = fminunc(f, d0, options);
        end
        
        function d_min = minimize_linearized_penalty_gd(obj, J, y, y_train, mu)
            % Minimizes the function loss(y + Jd) + 0.5*mu*norm(d)^2.
            % There is no closed form solution with NLL on softmax.
            
            %[d_min_mfun, fval] = check_linearized_penalty(obj, J, y, y_train, mu);
            
            % Row-major order.
            d = 0.1 * randn(size(J,2), 1);

            % Linesearch parameters.
            beta = 0.5;
            gamma = 10^-4;
            
            eps = 10^-1;

            i = 1;
            
            while true
                g = obj.gradient_linearized_penalty(d, J, y, y_train, mu);
                s = -g;
                
                if norm(s) <= eps
                   d_min = d;
                   break;
                end
                
                f = @(d) obj.linearized_penalty(d, J, y, y_train, mu);
                sigma = obj.armijo_linearized_penalty(f, d, g, s, beta, gamma);
                d = d + sigma * s;
                
                if mod(i,10) == 0
                    %disp(['NLLSoftmax linearized penalty: ', num2str(f(d)), ' gradnorm: ', num2str(norm(s))]);
                end
                
                i = i + 1;
            end
        end
        
        function sigma = armijo_linearized_penalty(~, f, d, g, s, beta, gamma)
            k = 1;
            
            while true
                sigma = (beta^k) / beta;
                
                if  f(d + sigma * s) - f(d) <= sigma * gamma * (g'*s)
                    break;
                end
                
                k = k + 1;
            end 
        end
        
        function [d_min, fval] = check_linearized_penalty(obj, J, y, y_train, mu)
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 5000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true, 'FiniteDifferenceType', 'central');
            
            d0 = ones(size(J,2), 1);
            
            f = @(d) obj.fun_linearized_penalty(d, J, y, y_train, mu);
            [d_min, fval] = fminunc(f, d0, options);
        end
        
        function [f, g] = fun_linearized_penalty(obj, d, J, y, y_train, mu)
            f = obj.linearized_penalty(d, J, y, y_train, mu);
            g = obj.gradient_linearized_penalty(d, J, y, y_train, mu);
        end
        
        function L = linearized_penalty(obj, d, J, y, y_train, mu)
            f = y(:) + J * d;
            f = reshape(f, size(y));
            
            L = obj.loss(f, y_train) + 0.5*mu*sum(d.^2);
        end
        
        function g = gradient_linearized_penalty(obj, d, J, y, y_train, mu)
            [c, N] = size(y);
            
            g = zeros(size(d));
            
            for i = 1:N
                from_sample = (i-1)*c + 1;
                to_sample = from_sample + c - 1;
                
                % (c,size(vec({W,b})).
                Ji = J(from_sample:to_sample,:);
                
                % f(x_i) with (c,1).
                yi = y(:,i);
                yi_train = y_train(:,i);
                fi = yi + Ji * d;
                
                % (c,1).
                gL = obj.gradient(fi, yi_train);
                
                g = g + Ji'*gL;
            end
            
            g = g + mu*d;
        end
    end
end