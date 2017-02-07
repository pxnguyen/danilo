function net = cnn_mv1m_init_language_model(varargin)
%CNN_IMAGENET_INIT_RESNET  Initialize the ResNet-50 model for ImageNet classification
% Args:
%   batch_size: the batch size to run at
%   num_frame: the sampled number of frames in a video
%   only_fc: if True, only train the last conv layer.

opts.classNames = {} ;
opts.classDescriptions = {} ;
opts.colorDeviation = zeros(3) ;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB
opts.pretrained_path = '';
opts.learning_schedule = [1e-5 * ones(1, 80000), 1e-6*ones(1, 80000), 1e-7*ones(1, 80000)];
opts.batch_size = 16;
opts.features = {'desc', 'cotags'};
opts = vl_argparse(opts, varargin) ;

net = dagnn.DagNN() ;

% -------------------------------------------------------------------------
%                                                           Meta parameters
% -------------------------------------------------------------------------

net.meta.normalization.imageSize = [1 1 300] ;
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

if exist(opts.pretrained_path, 'file')
  pretrained_model = load(opts.pretrained_path);
  pretrained_model = pretrained_model.net;
end

total_dim = 0;
for feature_index = 1:numel(opts.features)
  feature = opts.features{feature_index};
  switch feature
    case 'desc'
      total_dim = total_dim + 300;
    case 'cotags'
      total_dim = total_dim + 4000;
    case 'test'
      total_dim = total_dim + 4;
  end
end
out = numel(net.meta.classes.name);
% add the fc layer
fc_block = dagnn.Conv('size', [1, 1, total_dim, out],...
  'hasBias', true, ...
  'stride', 1, 'pad', 0);

net.addLayer('fc1', fc_block, ...
  'observed_input', 'fc1',...
  {'conv1_f', 'conv1_b'});

net.addLayer('fc2', fc_block, ...
  'corrupted_input',... % input
  'fc2',...
  {'conv1_f', 'conv1_b'});

if exist('pretrained_model', 'var')
  % load the params
  for i=1:numel(net.params)
    param_name = net.params(i).name;
    pre_param_index = 0;
    for j=1:numel(pretrained_model.params)
      if strcmp(param_name, pretrained_model.params(j).name)
        pre_param_index = j;
        net.params(i).value =  pretrained_model.params(pre_param_index).value;
      end
    end
    if pre_param_index == 0
      error('%s not found in pretrained', param_name);
    end
  end
else
  % init the params
  p = net.getParamIndex(net.layers(end).params) ;
  params = net.layers(end).block.initParams() ;
  params = cellfun(@gather, params, 'UniformOutput', false) ;
  [net.params(p).value] = deal(params{:}) ;
end

lName = net.layers(end).name;
net.addLayer('loss1', dagnn.Loss('loss', 'logistic2'), {'fc1', 'latent_label'}, 'loss1');
net.addLayer('loss2', dagnn.Loss('loss', 'logistic2'), {'fc2', 'observed_input'}, 'loss2');

% net.renameVar(net.vars(1).name, 'input');

net.rebuild()