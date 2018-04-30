classdef GDNetwork < Network
    methods
        function obj = GDNetwork(layers, h, dh, loss, X_train)
           obj@Network(layers, h, dh, loss, X_train);
        end
        
        function obj = train(obj, X_train, y_train, params)
            % params: Struct with
            %   linesearch parameters, iterations.
            
            for i = 1:params.iterations
                [obj, dW, db, L, ~] = obj.gradient(X_train, y_train);
                
                % Step directions.
                sW = cellfun(@(x) -x, dW, 'UniformOutput', 0);
                sb = cellfun(@(x) -x, db, 'UniformOutput', 0);
                
                if params.linesearch == 1
                    stepsize = params.stepsize;
                elseif params.linesearch == 2
                    stepsize = obj.armijo(obj.W, obj.b, dW, db, ...
                        sW, sb, X_train, y_train, params.beta, params.gamma);
                elseif params.linesearch == 3
                    stepsize = obj.pw(obj.W, obj.b, dW, db, ...
                        sW, sb, X_train, y_train, params.gamma, params.eta, params.beta);
                end
                
                if mod(i, 50) == 0
                    disp(['Loss: ', num2str(L), ', stepsize: ', num2str(stepsize), ...
                        ' gradnorm: ', num2str(norm(obj.to_vec(dW, db))) ,' (', num2str(i), ')']);
                end
                
                [obj.W, obj.b] = obj.update_weights(stepsize, obj.W, obj.b, sW, sb);
            end
            
            [obj, L, ~] = obj.f(X_train, y_train);
            disp(['Loss: ', num2str(L)]);
        end
    end
    
    methods(Access=private)
        function sigma = armijo(obj, W, b, dW, db, sW, sb, X_train, y_train, beta, gamma)
            k = 1;
            
            while 1
                sigma = (beta^k) / beta;
                
                if obj.check_armijo(sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    break;
                end
                
                k = k + 1;
            end
        end
        
        function r = check_armijo(obj, sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
            % Returns true if armijo condition is satisfied.
            
            % x_new = x + sigma * s
            [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
            
            % f(x_new) - f(x)
            [~, L_new, ~] = obj.f_eval(W_new, b_new, X_train, y_train, obj.loss);
            [~, L_current, ~] = obj.f_eval(W, b, X_train, y_train, obj.loss);
            f_new = L_new - L_current;
            
            slope = obj.directional_derivative(dW, db, sW, sb);
            
            if f_new <= sigma * gamma * slope
                r = true;
            else
                r = false;
            end
        end
        
        function sigma = pw(obj, W, b, dW, db, sW, sb, X_train, y_train, gamma, eta, beta)
            min_slope = eta * obj.directional_derivative(dW, db, sW, sb);
            
            sigma = 1;
            
            if obj.check_armijo(sigma, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                if obj.check_pw(sigma, W, b, sW, sb, X_train, y_train, min_slope)
                    return;
                else
                    sigma_pos = beta;
                    k = 1;
                    
                    while obj.check_armijo(sigma_pos, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                        k = k + 1;
                        sigma_pos = beta^k;
                    end
                    
                    sigma_neg = beta^(-1) * sigma_pos;
                end
            else
                sigma_neg = beta^(-1);
                k = 1;
                
                while ~obj.check_armijo(sigma_neg, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    k = k + 1;
                    sigma_neg = beta^(-k);
                end
                
                sigma_pos = beta * sigma_neg;
            end
            
            while ~obj.check_pw(sigma_neg, W, b, sW, sb, X_train, y_train, min_slope)
                s = (sigma_neg + sigma_pos) * 0.5;
                if obj.check_armijo(s, W, b, dW, db, sW, sb, X_train, y_train, gamma)
                    sigma_neg = s;
                else
                    sigma_pos = s;
                end
            end
            
            sigma = sigma_neg;
        end
        
        function r = check_pw(obj, sigma, W, b, sW, sb, X_train, y_train, min_slope)
            % Returns true if second Powell-Wolfe condition is satisfied.
            
            % x_new = x + sigma * s
            [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
            
            % Slope at new point.
            [~, dW_new, db_new, ~, ~] = obj.gradient_eval(W_new, b_new, X_train, y_train, obj.loss);
            slope = obj.directional_derivative(dW_new, db_new, sW, sb);
            
            if slope >= min_slope
                r = true;
            else
                r = false;
            end
        end
        
    end
    
end

