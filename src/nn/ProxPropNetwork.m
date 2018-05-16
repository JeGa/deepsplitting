classdef ProxPropNetwork < Network
    methods
        function obj = ProxPropNetwork(layers, h, dh, loss, X_train)
            obj@Network(layers, h, dh, loss, X_train);
        end
    end
    
    methods(Access=private)

    end
end

