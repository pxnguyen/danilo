function net = cnn_mv1m_init_resnet(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.averageImage = zeros(3,1) ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
opts.learning_schedule = [1e-5 * ones(1, 80000), 1e-6*ones(1, 80000), 1e-7*ones(1, 80000)];
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN.loadobj(load(opts.pretrained_path));

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.augmentation.jitterLocation = false ;
net.meta.augmentation.jitterFlip = false ;
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
net.meta.trainOpts.batchSize = 16 ;
net.meta.trainOpts.weightDecay = 0.0001 ;
%net.meta.trainOpts.numSubBatches = 4 ;

% -------------------------------------------------------------------------
%                                                           Removing layers
% -------------------------------------------------------------------------

% remove 'prob'
net.removeLayer(net.layers(end).name);

[h, w, in, out] = size(zeros(net.layers(end).block.size));
out = numel(net.meta.classes.name);

% cross-modality pretraining
%param_index = net.getParamIndex('conv1_filter')
%conv1_weights = net.params(param_index).value;
%conv1_weights_mean = mean(conv1_weights, 3)
%values = repmat(values, [1, 1, opts.num_flow * 2, 3]) %TODO(phucng): change 20 to num_flow
%net.params(param_index).value = values;
%net.layers(1).block.size[3] = opts.num_flow * 2; %TODO(phucng): change 20 to num_flow

% remove 'fc'
lName = net.layers(end).name;
net.removeLayer(net.layers(end).name);

pName = net.layers(end).name;
block = dagnn.Conv('size', [h, w, in, out], 'hasBias', true, ...
                   'stride', 1, 'pad', 0);
net.addLayer(lName, block, pName, lName, {[lName '_f'], [lName '_b']});
p = net.getParamIndex(net.layers(end).params) ;
params = net.layers(end).block.initParams() ;
params = cellfun(@gather, params, 'UniformOutput', false) ;
[net.params(p).value] = deal(params{:}) ;

% adding the 3D pooling
nFrames = 5;
i_pool5 = find(~cellfun('isempty', strfind({net.layers.name}, 'pool5')));
block = dagnn.Pooling3D() ;
block.method = 'max' ;
block.poolSize = [net.layers(i_pool5).block.poolSize nFrames];
block.pad = [net.layers(i_pool5).block.pad 0,0];
block.stride = [net.layers(i_pool5).block.stride 2];

% TODO(phucng): need to fix this
net.addLayerAt(i_pool5, 'pool3D5', block, ...
               [net.layers(i_pool5).inputs], ...
                 [net.layers(i_pool5).outputs]) ;
net.removeLayer('pool5') ;

lName = net.layers(end).name;
net.addLayer('loss', dagnn.Loss('loss', 'logistic'), {lName, 'label'}, 'objective');

% performance metrics
net.addLayer('sigmoid', dagnn.Sigmoid(), lName, 'sigmoid');
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
