classdef LLCNetwork < Network
    properties(Access=private)
        lambda % Lagrange multipliers for equality constraints.
        v % Network output (last layer linearities).
        rho % Weight for penalty term and dual update stepsize.
    end
    
    methods
        
        function obj = LLCNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);

            N = size(X_train, 2);
            c = layers(end);
            
            obj.lambda = ones(c, N);
            obj.v = 0.1 * randn(c, N);
            obj.rho = 1;
        end
        
        function obj = train(obj, X_train, y_train, params)
            % params: Struct with
            %   linesearch parameters, iterations.
            
            losses = zeros(params.iterations);
            
            for i = 1:params.iterations
                obj = obj.primal2_levmarq(X_train, y_train, params);
                
                obj = obj.primal1(X_train, y_train);
                
                obj = obj.dual(X_train);
                
                [obj, loss, constraint_norm, f, ~] = obj.lagrangian(X_train, y_train);
                disp(['Loss: ', num2str(loss), ', constraint: ', num2str(constraint_norm), ...
                    ', lagrangian: ', num2str(f), ' (', num2str(i), '/', num2str(params.iterations), ')']);
                losses(i) = loss;
                
                if mod(i, 50) == 0
                    title('Loss');
                    plot(1:i, losses(1:i));
                    drawnow;
                end
            end
        end
        
        function obj = check_gradients_primal2(obj, layers, X_train, y_train)
            x0 = obj.to_vec(obj.W, obj.b);
            
            options = optimoptions(@fminunc, 'MaxFunctionEvaluations', 20000, 'MaxIterations', 2000, ...
                'SpecifyObjectiveGradient', true, 'CheckGradients', true, 'FiniteDifferenceType', 'central');
            
            f = @(x) obj.fun_primal2(x, layers, X_train, y_train);
            [x, ~] = fminunc(f, x0, options);
            
            [W_min, b_min] = obj.to_mat(x, layers);
            
            obj.W = W_min;
            obj.b = b_min;
        end
    end
    
    methods(Access=private)
        function obj = primal1(obj, X_train, y_train)         
            [~, y] = obj.fp(X_train);
            
            obj.v = obj.loss.primal_update(y, obj.lambda, y_train, obj.rho);
        end
        
        function obj = primal2_gradientstep(obj, X_train, y_train, params)
            i = 0;
            max_iter = 1;
            
            while 1
                i = i + 1;
                
                % Gradient descent step w.r.t. ({W_j}, {b_j}).
                [~, ~, dW, db] = obj.primal2_gradient_eval(obj.W, obj.b, X_train, y_train);
                
                % Descent direction.
                sW = cellfun(@(x) -x, dW, 'un', 0);
                sb = cellfun(@(x) -x, db, 'un', 0);
                
                if params.linesearch == 1
                    stepsize = params.stepsize;
                elseif params.linesearch == 2
                    stepsize = obj.armijo_Wb(params, obj.W, obj.b, dW, db, sW, sb, X_train, y_train);
                end

                [obj.W, obj.b] = obj.update_weights(stepsize, obj.W, obj.b, sW, sb);
                
                % Gradient norm.
                gradnorm = sqrt(sum(cellfun(@(dW, db) sum([dW(:);db(:)].^2), dW, db)));
                            
                if gradnorm <= 10^-2 || stepsize <= 10^-6 || i == max_iter
                    break;
                end
            end
        end
        
        function obj = primal2_levmarq(obj, X_train, y_train, params)
            i = 0;
            max_iter = 1;
           
            % Damping factor.
            M = 0.001;
            factor = 10;
           
            while 1
                i = i + 1;
                
                [~, loss, ~, ~, ~] = obj.lagrangian(X_train, y_train);
                
                while 1
                    [W_new, b_new] = obj.levmarq_step(obj.W, obj.b, X_train, M);
                    
                    [~, loss_new, ~, ~, ~] = obj.lagrangian_eval(W_new, b_new, obj.lambda, obj.v, X_train, y_train);
                
                    if loss < loss_new
                       M = M * factor;
                    else
                       obj.W = W_new;
                       obj.b = b_new;
                       
                       M = M / factor;
                       break;
                    end
                end
                
                if i == max_iter
                    break; 
                end
            end
        end
        
        function [W_new, b_new] = levmarq_step(obj, W, b, X_train, M)
            [obj, dW, db, ~] = obj.jacobian_eval_noloss(W, b, X_train);

            [~, y] = obj.fp_eval(W, b, X_train);
            r = -obj.lambda(:)/obj.rho + obj.v(:) -y(:);

            L = size(dW, 2);
            W_new = cell(1, L);
            b_new = cell(1, L);

            solve_step = @(J, r, M) (J'*J + M * eye(size(J, 2))) \ J'*r;

            for j = 1:L
                sW = solve_step(dW{j}, r, M);
                sb = solve_step(db{j}, r, M);

                W_new{j} = W{j} + reshape(sW, size(W{j}, 2), size(W{j}, 1))';
                b_new{j} = b{j} + reshape(sb, size(b{j}));
            end
        end
        
        function stepsize = armijo_Wb(obj, params, W, b, dW, db, sW, sb, X_train, y_train)
            k = 1;
            
            while 1
                sigma = (params.beta^k) / params.beta;

                % x_new = x + sigma * s
                [W_new, b_new] = obj.update_weights(sigma, W, b, sW, sb);
                
                % f(x_new) - f(x)
                [~, ~, ~, L_new, ~] = obj.lagrangian_eval(W_new, b_new, obj.lambda, obj.v, X_train, y_train);
                [~, ~, ~, L_current, ~] = obj.lagrangian_eval(W, b, obj.lambda, obj.v, X_train, y_train);
                f_new = L_new - L_current;

                slope = obj.directional_derivative(dW, db, sW, sb);

                if f_new <= sigma * params.gamma * slope
                    stepsize = sigma;
                    break;
                end

                k = k + 1;
            end
        end
        
        function obj = dual(obj, X_train)
            % Gradient ascent dual update.
            % Forward pass with new weights, bias.
            [obj, y] = obj.fp(X_train);
            
            % Dual update with new weights, bias and v.
            obj.lambda = obj.lambda + obj.rho * (y - obj.v);
        end
        
        function [obj, loss, constraint_norm, f, y] = lagrangian(obj, X_train, y_train)
            [obj, loss, constraint_norm, f, y] = obj.lagrangian_eval(obj.W, obj.b, obj.lambda, obj.v, X_train, y_train);
        end
        
        function [obj, loss, constraint_norm, f, y] = lagrangian_eval(obj, W, b, lambda, v, X_train, y_train)
            [obj, y] = obj.fp_eval(W, b, X_train);
            
            loss = obj.loss.loss(v, y_train);
            constraint = y(:) - v(:);
            constraint_norm = sum(constraint.^2);
            
            % TODO: Regularizer.
            % nu = 1;
            reg = 0; %0.5 * nu * sum(x.^2);
            
            f = loss + lambda(:)' * constraint + (obj.rho/2) * constraint_norm + reg;
        end

        function [obj, L, dW, db] = primal2_gradient_eval(obj, W, b, X_train, y_train)
            % For primal2 v and lambda are fixed.
            [~, ~, ~, L, ~]  = obj.lagrangian_eval(W, b, obj.lambda, obj.v, X_train, y_train);
            
            % The gradient of the lagrangian is equal to the gradient of the reformulated squared error so we 
            % can use the gradient function with the LeastSquares error.
            % Note: The function gradient_eval does again a forward pass so
            % this could be optimized to just do one forward pass.
            r = obj.v - (1/obj.rho) .* obj.lambda;
            [~, dW, db, ~, ~] = obj.gradient_eval(W, b, X_train, r, LeastSquares(obj.rho));
        end

        function [L, g] = fun_primal2(obj, x, layers, X_train, y_train)
            [W, b] = obj.to_mat(x, layers);
            
            [obj, L, dW, db] = obj.primal2_gradient_eval(W, b, X_train, y_train);
            
            g = obj.to_vec(dW, db);
        end
        
    end
    
end

