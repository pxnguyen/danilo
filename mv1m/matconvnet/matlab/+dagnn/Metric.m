classdef Metric < dagnn.ElementWise
  properties
    top_k = 1
    num_false_positives = 0
    num_true_positives = 0
  end

  properties (Transient)
    average = 0
    numAveraged = 0
  end

  methods
    function outputs = forward(obj, inputs, params)
      scores = inputs{1};
      labels = inputs{2};
      [tp, fp] = stream_precision_at_k(scores, labels, obj.top_k);
      obj.num_true_positives = obj.num_true_positives + tp;
      obj.num_false_positives = obj.num_false_positives + fp;
      outputs{1} = obj.num_true_positives/(obj.num_true_positives+obj.num_false_positives);
      n = obj.numAveraged ;
      m = n + size(inputs{1},4) ;
      obj.average = (n * obj.average + gather(outputs{1})) / m ;
      obj.numAveraged = m ;
    end

    function [derInputs, derParams] = backward(obj, inputs, params, derOutputs)
      derInputs{1} = zeros(size(inputs{1}));
      derInputs{2} = [] ;
      derParams = {} ;
    end

    function reset(obj)
      obj.average = 0 ;
      obj.numAveraged = 0 ;
    end

    function outputSizes = getOutputSizes(obj, inputSizes, paramSizes)
      outputSizes{1} = [1 1 1 inputSizes{1}(4)] ;
    end

    function rfs = getReceptiveFields(obj)
      % the receptive field depends on the dimension of the variables
      % which is not known until the network is run
      rfs(1,1).size = [NaN NaN] ;
      rfs(1,1).stride = [NaN NaN] ;
      rfs(1,1).offset = [NaN NaN] ;
      rfs(2,1) = rfs(1,1) ;
    end

    function obj = Metric(varargin)
      obj.load(varargin) ;
    end
  end
end
