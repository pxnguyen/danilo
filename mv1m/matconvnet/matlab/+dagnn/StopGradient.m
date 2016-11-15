classdef StopGradient < dagnn.ElementWise
  methods
    function outputs = forward(obj, inputs, params)
      outputs = inputs;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs = {} ;
      for i=1:numel(inputs)
        derInputs{end+1} = [];
      end
      derParams = {} ;
    end
  end
end
