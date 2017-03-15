function net = cnn_mv1m_init_resnet(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification
% Args:
%   batch_size: the batch size to run at
%   num_frame: the sampled number of frames in a video
%   only_fc: if True, only train the last conv layer.

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
opts.learning_schedule = [1e-5 * ones(1, 80000), 1e-6*ones(1, 80000), 1e-7*ones(1, 80000)];
opts.batch_size = 16;
opts.num_frame = 5;
opts.only_fc = true;
opts.stop_gradient_location = 'res5b';
opts.dropout_ratio = 0.0;
opts.loss_type = 'logistic';
opts.add_fc128 = false;
opts.input_type = 'video';
opts = vl_argparse(opts, varargin) ;

if opts.only_fc
  if strcmp(opts.input_type, 'video')
    opts.stop_gradient_location = 'pool3D5';
  else
    opts.stop_gradient_location = 'pool5';
  end
end

net = dagnn.DagNN.loadobj(load(opts.pretrained_path));

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.augmentation.jitterLocation = true ;
net.meta.augmentation.jitterFlip = true ;
net.meta.augmentation.jitterScale = true ;
%net.meta.augmentation.rgbVariance = zeros(0, 3);

net.meta.normalization.imageSize = [224 224 3] ;
net.meta.normalization.cropSize = net.meta.normalization.imageSize(1) / 256 ;
net.meta.normalization.averageImage = opts.averageImage ;

net.meta.classes.name = opts.classNames ;

net.meta.inputSize = {'input', [net.meta.normalization.imageSize 32]} ;

lr = opts.learning_schedule;
net.meta.trainOpts.learningRate = lr ;
net.meta.trainOpts.numEpochs = numel(lr) ;
net.meta.trainOpts.momentum = 0.9 ;
net.meta.trainOpts.batchSize = opts.batch_size ;
net.meta.trainOpts.weightDecay = 0.0001 ;
%net.meta.trainOpts.numSubBatches = 4 ;

% -------------------------------------------------------------------------
%                                                           Removing layers
% -------------------------------------------------------------------------

% cross-modality pretraining
%param_index = net.getParamIndex('conv1_filter')
%conv1_weights = net.params(param_index).value;
%conv1_weights_mean = mean(conv1_weights, 3)
%values = repmat(values, [1, 1, opts.num_flow * 2, 3]) %TODO(phucng): change 20 to num_flow
%net.params(param_index).value = values;
%net.layers(1).block.size[3] = opts.num_flow * 2; %TODO(phucng): change 20 to num_flow

% remove 'prob'
net.removeLayer('prob');

% turn all the batch normalization off
for iparams = 1:length(net.params)
  if strfind(net.params(iparams).name, 'bn')
    net.params(iparams).learningRate = 0;
    net.params(iparams).weightDecay = 0;
  end
end

% add the dropout layer
if opts.dropout_ratio > 0
  index_res5c_relu = find(arrayfun(@(x) strcmp(x.name,'res5c_relu'), net.layers)) ;
  dropout_block = dagnn.DropOut() ;
  dropout_block.rate = opts.dropout_ratio ;
  dropout_layer_name = ['drop_' net.layers(index_res5c_relu).name];

  net.addLayerAt(index_res5c_relu, dropout_layer_name,...
    dropout_block, ...
    net.layers(index_res5c_relu).outputs, ...
    dropout_layer_name) ;

  i_pool5 = find(~cellfun('isempty', strfind({net.layers.name}, 'pool5')));
  net.layers(i_pool5).inputs{1} = dropout_layer_name;
end

if strcmp(opts.input_type, 'video')
  % adding the 3D pooling
  i_pool5 = find(~cellfun('isempty', strfind({net.layers.name}, 'pool5')));
  block = dagnn.Pooling3D() ;
  block.method = 'max' ;
  block.poolSize = [net.layers(i_pool5).block.poolSize opts.num_frame];
  block.pad = [net.layers(i_pool5).block.pad 0,0];
  block.stride = [net.layers(i_pool5).block.stride 2];

  net.addLayerAt(i_pool5, 'pool3D5', block, ...
    [net.layers(i_pool5).inputs], ...
    'pool3D5') ;
  net.removeLayer('pool5') ;
end

% remove 'prob'
index_fc1000 = find(arrayfun(@(x) strcmp(x.name, 'fc1000'), net.layers));
fc1k_tobe_removed = net.layers(index_fc1000);
[h, w, in, ~] = size(zeros(net.layers(index_fc1000).block.size));
net.removeLayer(net.layers(index_fc1000).name);
out = numel(net.meta.classes.name);
% prev_layer = fc1k_tobe_removed.inputs{1};

% re-add the fc layer
fc_block = dagnn.Conv('size', [h, w, in, out], 'hasBias', true, ...
  'stride', 1, 'pad', 0);

if strcmp(opts.input_type, 'video')
  net.addLayer('fc1000', fc_block, ...
    'pool3D5', 'fc1000',...
    {['fc1000_f'], ['fc1000_b']});
else
  net.addLayer('fc1000', fc_block, ...
    'pool5', 'fc1000',...
    {['fc1000_f'], ['fc1000_b']});
end
% init the new layer params
p = net.getParamIndex(net.layers(end).params) ;
params = net.layers(end).block.initParams() ;
params = cellfun(@gather, params, 'UniformOutput', false) ;
[net.params(p).value] = deal(params{:}) ;

index_fc1000 = find(arrayfun(@(x) strcmp(x.name, 'fc1000'), net.layers));
lName = net.layers(index_fc1000).name;

switch opts.loss_type
  case 'logistic'
    net.addLayer('loss', dagnn.Loss('loss', 'logistic'), {lName, 'label'}, 'objective');
  case 'logistic2'
    net.addLayer('loss', dagnn.Loss('loss', 'logistic2'), {lName, 'label'}, 'objective');
  case 'softmax'
    net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {lName, 'label'}, 'objective') ;
  case 'L2'
    net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {lName, 'label'}, 'objective') ;
  otherwise
      error('Unrecognized loss type: %s', opts.loss_type);
end

% performance metrics
net.addLayer('sigmoid', dagnn.Sigmoid(), lName, 'sigmoid');

% add the stop gradient block
stop_gradient_block = dagnn.StopGradient();
layer_before_stop_index = find(arrayfun(@(x) strcmp(x.name, opts.stop_gradient_location), net.layers)) ;
layer_before_stop = net.layers(layer_before_stop_index);
outputs = layer_before_stop.outputs;
if numel(outputs)>1; error('This case is not handled\n'); end;
bstop_output = outputs{1};
for i=1:layer_before_stop_index
  params = net.layers(i).params;
  for i_param=1:numel(params)
    param_name = params{i_param};
    param_index = find(arrayfun(@(x) strcmp(x.name, param_name), net.params)) ;
    net.params(param_index).learningRate = 0;
    net.params(param_index).weightDecay = 0;
  end
end

net.addLayerAt(layer_before_stop_index, 'stop_gradient', stop_gradient_block,...
 opts.stop_gradient_location, 'stop_gradient');

% find the layer with input matches bstop_output
for i_layer=1:numel(net.layers)
  if strcmp(net.layers(i_layer).inputs{1}, bstop_output) &&...
      ~strcmp(net.layers(i_layer).name, 'stop_gradient')
    if numel(net.layers(i_layer).inputs) > 1
      error('This case is not handled\n');
    end
    net.layers(i_layer).inputs{1} = 'stop_gradient';
  end
end

net.addLayer('error',...
  dagnn.Loss('loss', 'hit@k', 'opts', {'topK', 1}),...
  {'sigmoid', 'label'}, 'hit_at_1') ;


% net.addLayer('error5',...
%   dagnn.Loss('loss', 'hit@k', 'opts', {'topK', 5}),...
%   {'sigmoid', 'label'}, 'hit_at_5') ;

%trainable_layers = {'pool3D5', 'loss', lName};
%for layer_index = 1:length(net.layers)
%  if ismember(net.layers(layer_index).name, trainable_layers)
%    net.layers(layer_index).trainable = true;
%  else
%    net.layers(layer_index).trainable = false;
%  end
%end

%net.addLayer('softmax', dagnn.SoftMax(), lName, 'softmax');

net.renameVar(net.vars(1).name, 'input');

net.rebuild()

% if opts.only_fc
%   % add the stop gradient layer
% 
%   for iparams = 1:length(net.params)
%     net.params(iparams).learningRate = 0;
%     net.params(iparams).weightDecay = 0;
%   end
% end

% add the fc128 layer
% if opts.add_fc128
%   fc_block = dagnn.Conv('size', [h, w, in, 128], 'hasBias', true, ...
%     'stride', 1, 'pad', 0);
%   net.addLayer('fc128', fc_block, ...
%     prev_name, 'fc128',...
%     {'fc128_f', 'fc128_b'});
% 
%   % init weights
%   p = net.getParamIndex(net.layers(end).params) ;
%   params = net.layers(end).block.initParams() ;
%   params = cellfun(@gather, params, 'UniformOutput', false) ;
%   [net.params(p).value] = deal(params{:}) ;
% 
%   prev_name = 'fc128';
%   in = 128;
% end