% TODO: Do not use matlab fminunc for nll minimize_linearized_penalty.

classdef ProxDescentNetwork < Network   
    methods
        function obj = ProxDescentNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);
        end
        
        function [obj, losses] = train(obj, X_train, y_train, params)
            % params: Struct with
            %   iterations, tau, sigma, mu_min.
            
            mu = params.mu_min;
            
            losses = zeros(1, params.iterations);
            
            [~, L_start, ~] = obj.f(X_train, y_train);
            start_loss = L_start;
            
            disp(['ProxDescent: Loss: ', num2str(start_loss)]);
            
            for i = 1:params.iterations
                [obj, J, y] = obj.jacobian_noloss_matrix(X_train);
                
                while true
                    % Compute step (minimum of linearization).
                    d = obj.loss.minimize_linearized_penalty(J, y, y_train, mu);
                    
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
                
                disp(['ProxDescent: Loss: ', num2str(L_new), ' (', num2str(i), '/', num2str(params.iterations), ')']);
                
                losses(i) = L_new;
            end
            
            losses = [start_loss,losses];
        end
    end
    
    methods(Access=private)        
        function [obj, L] = loss_linearized(obj, y_train, J, y, d, mu)
            % h(c(x) + grad(c(x))*d) + 0.5 * mu * norm(d)^2.
            % d: Vectorized weight and biases with
            % (weights in row-major layer 1 to L, biases layer 1 to L).
            [c, N] = size(y_train);
            lin = y + reshape(J*d, c, N);
            
            loss = obj.loss.loss(lin, y_train);
            
            % Regularizer.
            reg = 0.5 * mu * norm(d)^2;
            
            L = loss + reg;
        end
    end
end

