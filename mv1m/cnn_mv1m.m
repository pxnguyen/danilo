function [net, info] = cnn_mv1m(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet

opts.dataDir = '/mnt/large/pxnguyen/vine-large-2/';
opts.modelType = 'resnet-50' ;
opts.network = [] ;
opts.networkType = 'simplenn' ;
opts.batchNormalization = true ;
opts.weightInitMethod = 'gaussian' ;
[opts, varargin] = vl_argparse(opts, varargin) ;

sfx = opts.modelType ;
if opts.batchNormalization, sfx = [sfx '-bnorm'] ; end
sfx = [sfx '-' opts.networkType] ;
opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/danilo')
%opts.expDir = fullfile(vl_rootnn, 'data', ['imagenet12-' sfx]) ;
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 12 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.train = struct() ;
opts.frame_dir = '/tmp/vine-images';
opts.iter_per_epoch = 80000;
opts.iter_per_save = 2000;
opts.num_eval_per_epoch = 8000;
opts.pretrained_path = '';
opts.learning_schedule = 0;
opts.num_frame = 10;
opts.batch_size = 9;
opts.only_fc = false;
opts.dropout_ratio = 0;
opts.label_type = 'original';
opts.loss_type = 'logistic';
opts.input_type = 'video';
opts.add_fc128 = false;
opts.train = struct();
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'videos');
else
  imdb = cnn_mv1m_setup_data('dataDir', opts.dataDir) ;
  mkdir(opts.expDir) ;
  save(opts.imdbPath, '-struct', 'imdb') ;
end

if exist(fullfile(opts.expDir, 'latent_labels.mat'), 'file')
  fprintf('Loading latent labels...\n');
  latent_labels = load(fullfile(opts.expDir, 'latent_labels.mat'));
  fprintf('Done\n');
  imdb.latent_labels = latent_labels;
end

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
else
  train = find(imdb.images.set == 1) ;
  images = fullfile(imdb.imageDir, imdb.images.name(train(1:100:end))) ;
  [averageImage, rgbMean, rgbCovariance] = getImageStats(images, ...
    'imageSize', [256 256], ...
    'input_type', opts.input_type,...
    'numThreads', opts.numFetchThreads, ...
    'frame_dir', opts.frame_dir,...
    'gpus', opts.train.gpus) ;
  save(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
end
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_mv1m_init_resnet('averageImage', rgbMean, ...
        'colorDeviation', rgbDeviation, ...
        'pretrained_path', opts.pretrained_path, ...
        'learning_schedule', opts.learning_schedule, ...
        'batch_size', opts.batch_size, ...
        'num_frame', opts.num_frame, ...
        'only_fc', opts.only_fc, ...
        'dropout_ratio', opts.dropout_ratio,...
        'loss_type', opts.loss_type,...
        'input_type', opts.input_type,...
        'add_fc128', opts.add_fc128,...
        'classNames', imdb.classes.name);
      opts.networkType = 'dagnn' ;
  end
else
  net = opts.network ;
  opts.network = [] ;
end

% -------------------------------------------------------------------------
%                                                                     Learn
% -------------------------------------------------------------------------

switch opts.networkType
  case 'simplenn', trainFn = @cnn_train ;
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  'iter_per_epoch', opts.iter_per_epoch,...
  'iter_per_save', opts.iter_per_save,...
  'num_eval_per_epoch', opts.num_eval_per_epoch, ...
  'prefetch', true, ...
  'nesterovUpdate', true, ...
  'label_type', opts.label_type, ...
  'loss_type', opts.loss_type,...
  net.meta.trainOpts, ...
  opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
    meta.normalization.imageSize(1:2)) ;
end

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct(...
  'useGpu', useGpu, ...
  'numThreads', opts.numFetchThreads, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

bopts.frame_dir = opts.frame_dir;
bopts.num_frame = opts.num_frame;
bopts.label_type = opts.label_type;
bopts.loss_type = opts.loss_type;
bopts.input_type = opts.input_type;

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

fn = @(x, y) getBatch(bopts, useGpu, lower(opts.networkType), x, y);

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
if strcmp(opts.input_type, 'video')
  images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
else
  images = strcat([imdb.image_path filesep], imdb.images.name(batch)) ;
end
if isempty(images)
  return
end
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
video_paths = images;
all_files = cell(length(video_paths),1);
if strcmp(opts.input_type, 'video')
  for video_index = 1:length(video_paths)
    files = extract_frames(video_paths{video_index}, 'dest_dir', opts.frame_dir);
    if strcmp(phase, 'train')
      frame_selection = randperm(length(files));
      frame_selection = frame_selection(1:opts.num_frame);
    elseif strcmp(phase, 'test')
      frame_selection = floor(linspace(1, length(files), opts.num_frame));
    end
    all_files{video_index} = files(frame_selection);
  end
  all_files = cat(2, all_files{:});
else
  for video_index = 1:length(video_paths)
    video_name = sprintf('%s.jpg', video_paths{video_index});
    all_files{video_index} = video_name;
  end
end
data = getImageBatch(all_files, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  switch opts.loss_type
    case 'logistic'
      labels = double(full(imdb.images.label(:, batch)));
      labels(labels==0) = -1;
      labels = permute(labels, [3, 4, 1, 2]);
    case 'logistic2'
      labels = double(full(imdb.images.label(:, batch)));
      labels = permute(labels, [3, 4, 1, 2]);
    case 'softmax'
      labels = double(full(imdb.images.label(batch)));
    otherwise
      error('Unrecognized loss type: %s', opts.loss_type);
  end
  
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end