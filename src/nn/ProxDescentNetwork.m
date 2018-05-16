classdef ProxDescentNetwork < Network   
    methods
        function obj = ProxDescentNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);
            
            if ~isa(loss, 'LeastSquares')
                error('Currently only works with least squares error.');
            end
        end
        
        function obj = train(obj, X_train, y_train, params)
            % params: Struct with
            %   iterations, tau, sigma, mu_min.
            
            mu = params.mu_min;
            
            for i = 1:params.iterations
                [obj, J, y] = obj.jacobian_noloss_matrix(X_train);
                
                while true
                    % Compute step (minimum of linearization).
                    [obj, d] = obj.minimize_ls_penalty(y_train, mu, J, y);
                    
                    % Check regularizer weight.
                    [~, L_current, ~] = obj.f(X_train, y_train);
                    
                    [sW, sb] = obj.to_mat(d, obj.layers, 1);
                    [W_new, b_new] = obj.update_weights(1, obj.W, obj.b, sW, sb);
                    [~, L_new, ~] = obj.f_eval(W_new, b_new, X_train, y_train, obj.loss);
                    
                    [~, L_new_linearized] = obj.loss_linearized(y_train, J, y, d, mu);
                    
                    diff_real = L_current - L_new;
                    diff_linearized = L_current - L_new_linearized;
                    
                    if diff_real >= params.sigma * diff_linearized
                        % Make mu smaller for next step.
                        mu = max(params.mu_min, mu/params.tau);
                        break
                    else
                        mu = params.tau * mu;
                    end
                end
                
                obj.W = W_new;
                obj.b = b_new;
                
                disp(['Loss: ', num2str(L_new), ' (', num2str(i), '/', num2str(params.iterations), ')']);
            end
        end
    end
    
    methods(Access=private)
        function [obj, d] = minimize_ls_penalty(obj, y_train, mu, J, y)
            N = 1/obj.loss.C;
            
            vec = @(x) x(:);
            
            d = (mu*N*eye(size(J, 2)) + J'*J)^(-1)*J'*vec(y_train - y);
        end
        
        function [obj, J, y] = jacobian_noloss_matrix(obj, X_train)
            % Jc(x) (Jacobian with row-major vectorization) and c(x).
            [obj, dW, db, y] = obj.jacobian_eval_noloss(obj.W, obj.b, X_train);
            
            % Build Jacobian.
            JW = [dW{:}];
            Jb = [db{:}];
            J = [JW, Jb];
        end
        
        function [obj, L] = loss_linearized(obj, y_train, J, y, d, mu)
            % h(c(x) + grad(c(x))*d) + 0.5 * mu * norm(d)^2.
            % d: Vectorized weight and biases with
            % (weights in row-major layer 1 to L, biases layer 1 to L).
            [N, c] = size(y_train);
            lin = y + reshape(J*d, N, c);
            
            loss = obj.loss.loss(lin, y_train);
            
            % Regularizer.
            reg = 0.5 * mu * norm(d)^2;
            
            L = loss + reg;
        end
    end
end

