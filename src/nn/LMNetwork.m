classdef LMNetwork < Network
    methods
        function obj = LMNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);
            
            if ~isa(loss, 'LeastSquares')
                error('Only works with least squares error.');
            end
        end
        
        function [obj, losses] = train(obj, X_train, y_train, params)
            % params: Struct with
            % Damping parameter, iterations.
            
            M = params.M;
            factor = params.factor;
            
            losses = zeros(1, params.iterations);
            
            [~, L_start, ~] = obj.f(X_train, y_train);
            start_loss = L_start;
            
            disp(['LM: Loss: ', num2str(start_loss)]);
            
            for i = 1:params.iterations
                [~, dW, db, y] = obj.jacobian_eval_noloss(obj.W, obj.b, X_train);
                J = obj.to_jacobian(dW, db);
                L = obj.loss.loss(y, y_train);
                
                while true
                    [W_new, b_new] = obj.levmarq_step(obj.W, obj.b, J, y, y_train, M);
                    
                    [~, L_new, ~] = obj.f_eval(W_new, b_new, X_train, y_train, obj.loss);
                    
                    if L < L_new
                        M = M * factor;
                    else
                        obj.W = W_new;
                        obj.b = b_new;
                        
                        M = M / factor;
                        break;
                    end
                end
                
                disp(['LM: Loss: ', num2str(L_new), ' (', num2str(i), '/', num2str(params.iterations), ')']);
                
                losses(i) = L_new;
            end
            
            losses = [start_loss,losses];
        end
    end
    
    methods(Access=private)
        function [W_new, b_new] = levmarq_step(obj, W, b, J, y, y_train, M)
            r = y_train(:) - y(:);
            
            s = (J'*J + M*eye(size(J, 2))) \ J'*r;
            
            [sW, sb] = obj.to_mat(s, obj.layers, 1);
            [W_new, b_new] = obj.update_weights(1, W, b, sW, sb);
        end
    end
end

