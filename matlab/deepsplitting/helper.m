classdef helper
    % Because matlab does not allow namespaces ...
    
    methods(Static)
        function params = get_params(p)
            if strcmp(p, 'GD')
                % 1 = fixed stepsize, 2 = Armijo, 3 = Powell-Wolfe.

                params.linesearch = 1;
                params.stepsize = 5e-2;

                %params.linesearch = 2;
                %params.beta = 0.5;
                %params.gamma = 10^-4;

                %params.linesearch = 3;
                %params.gamma = 10^-4;
                %params.eta = 0.7;
                %params.beta = 4;
                
                params.iterations = 2000;
            elseif strcmp(p, 'LLC')
                % 1 = fixed stepsize, 2 = Armijo.

                %params.linesearch = 1;
                %params.stepsize = 0.001;

                %params.linesearch = 2;
                %params.beta = 0.5;
                %params.gamma = 10^-4;

                % LM damping factor.
                params.M = 0.001;
                params.factor = 10;
                
                params.iterations = 100;
            elseif strcmp(p, 'ProxDescent')
                % Regularizer weight multiplier.
                params.tau = 1.5;
                % Step quality multiplier.
                params.sigma = 0.5;
                % Initial regularizer weight.
                params.mu_min = 0.3;
                
                params.iterations = 18;
            elseif strcmp(p, 'LM')
                % LM Damping factor.
                params.M = 0.001;
                params.factor = 10;
                
                params.iterations = 100;
            elseif strcmp(p, 'ProxProp')
                %params.tau = 1;
                params.tau = 0.005;
                params.tau_theta = 10;
                
                params.iterations = 100;
            else
                error('Unknown algorithm parameter.');
            end
            
            %params.iterations = 50;
        end

        function [X_train, y_train, X_test, y_test, dim, classes] = get_data(type, data_type, do_plot, ptrain)
            addpath('datasets');

            if type == 1
                switch data_type
                    case 'corners'
                        data = corners();
                    case 'outlier'
                        data = outlier();
                    case 'halfkernel'
                        data = halfkernel();
                    case 'moon'
                        data = crescentfullmoon();
                    case 'clusters'
                        data = clusterincluster();
                    case 'spirals'
                        data = twospirals(250, 360, 90, 1.2);
                end
            elseif type == 2
                if strcmp(data_type, 'reg_sinus')
                    x = linspace(0, 2*pi, 30);
                    y = sin(x) + 0.1 * randn(size(x));
                    data = [x; y]';
                end
            else
               error('Unsupported type.'); 
            end

            dim = size(data, 2)-1;

            % Shuffle data.
            shuffle = randsample(1:size(data, 1), size(data, 1));
            X = data(shuffle, 1:dim)';

            if type == 2
                % Regression.
                y = data(shuffle, dim+1);
            else
                % Classification.
                y = data(shuffle, dim+1)+1;
            end

            % Divide data into training and test set.
            n = uint64(ptrain*size(data, 1));

            X_train = X(:, 1:n);
            y_train = y(1:n, :);
            X_test = X(:, n+1:end);
            y_test = y(n+1:end, :);

            if type == 2
                classes = 1;
                y_train = y_train';
                y_test = y_test';
            else
                % Classification
                y_train = helper.one_hot(y_train);
                y_test = helper.one_hot(y_test);
                classes = max(y);
            end

            if do_plot
                if dim == 2
                    figure(1);
                    scatter(data(:,1), data(:,2), 10, data(:,3));
                    axis equal;
                    title('Ground truth');
                    drawnow
                elseif dim == 1
                    figure(1);
                    scatter(data(:,1), data(:,2));
                    title('Ground truth');
                    drawnow 
                end
            end
        end

        function [h, dh] = activation_function(type)
            if type == 1
                h = @(t) 1 ./ (1 + exp(-t));
                dh = @(t) h(t) .* (1 - h(t));
            elseif type == 2
                h = @(t) max(0, t);
                dh = @(t) 1 * (t>0);
            end
        end

        function plot_result_cls(X_train, y, f)
            [~,C] = max(y, [], 1);
            C = C - 1;

            % y is of shape (cls, samples).
            figure(f);
            scatter(X_train(1,:), X_train(2,:), 10, C);
            axis equal;
            drawnow
        end

        function plot_result_reg(X_train, y, f)
            figure(f);
            scatter(X_train, y);
            drawnow
        end

        function x_onehot = one_hot(x)
            % x: (N,1).
            classes = max(x);

            x_onehot = zeros(classes, size(x,1));
            ind = sub2ind(size(x_onehot), x', 1:size(x',2));
            x_onehot(ind) = 1;
        end

        function plot_grid(network)
            [X, Y] = meshgrid(-10:0.1:10, -10:0.1:10);
            X_input = [X(:), Y(:)]';

            [~, y] = network.fp(X_input);
            y = Softmax.softmax(y);

            xsize = size(X,2);
            ysize = size(X,1);

            [~, C] = max(y, [], 1);
            C = C - 1;
            y = reshape(C, ysize, xsize);

            figure(5);
            axis equal;
            image(y', 'CDataMapping', 'scaled')
        end

        function misclassified = results_cls(y, y_gt)        
            [~, y_ind] = max(y);
            [~, y_gt_ind] = max(y_gt);

            misclassified = sum(y_ind ~= y_gt_ind);
        end

        function y = predict_nllsm(network, X)
            [~, y] = network.fp(X);
            y = Softmax.softmax(y);
        end

        function y = predict_ls(network, X)
            [~, y] = network.fp(X);
        end
        
        function save_to_csv(X_train, y_train, X_test, y_test, name)
            csvwrite(strcat(name, '_X_train'), X_train);
            csvwrite(strcat(name, '_y_train'), y_train);
            csvwrite(strcat(name, '_X_test'), X_test);
            csvwrite(strcat(name, '_y_test'), y_test);
        end
    end
end

