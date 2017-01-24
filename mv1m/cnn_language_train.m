function [net, info] = cnn_language_train(varargin)
%CNN_IMAGENET   Train a CNN on w2v inputs

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

opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.batch_size = 200;
opts.iter_per_epoch = 80000;
opts.iter_per_save = 2000;
opts.learning_schedule = 0;
opts.train = struct() ;
opts.features = {'desc', 'cotags'};
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

if exist(opts.imdbPath)
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'videos');
end
% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_mv1m_init_language_model(...
        'learning_schedule', opts.learning_schedule, ...
        'batch_size', opts.batch_size, ...
        'features', opts.features, ...
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
  case 'dagnn', trainFn = @cnn_train_dag ;
end

[net, info] = trainFn(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  'iter_per_epoch', opts.iter_per_epoch,...
  'iter_per_save', opts.iter_per_save,...
  'prefetch', true, ...
  'nesterovUpdate', true, ...
  net.meta.trainOpts, ...
  opts.train) ;

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

useGpu = numel(opts.train.gpus) > 0 ;

bopts.test = struct('useGpu', useGpu);

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
bopts.features = opts.features;
fn = @(x, y) getBatch(bopts, useGpu, lower(opts.networkType), x, y);

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if isempty(images); return; end;

w2v_storage = '/home/phuc/Research/word2vec_storage';
video_paths = images;

augmented_labels = imdb.images.label(:, batch);
data = cell(numel(opts.features), 1);
for feature_index = 1:numel(opts.features)
  feature = opts.features{feature_index};
  if strcmp(feature, 'desc')
    w2v_data = cell(length(video_paths), 1);
    for video_index = 1:length(video_paths)
      video_path = video_paths{video_index};
      [~, name, ~] = fileparts(video_path);
      w2v_path = fullfile(w2v_storage, sprintf('%s.mat', name));

      if exist(w2v_path, 'file')
        lstruct = load(w2v_path);
        if ~isempty(lstruct.all_vecs)
          w2v_data{video_index} = sum(lstruct.all_vecs, 1);
        else
          w2v_data{video_index} = rand(1, 300);
        end
      else
        w2v_data{video_index} = rand(1, 300);
      end
    end
    w2v_data = cat(1, w2v_data{:});
    data{feature_index} = w2v_data';
  elseif strcmp(feature, 'cotags')
    cotags_data = full(imdb.images.label(:, batch));
    rand_noise = rand(size(cotags_data));
    prob = 0.5;
    for video_index = 1:length(video_paths)
      %indeces = cotags_data(:, video_index);
      rand_noise_vid = rand_noise(:, video_index);
      to_flip_off = rand_noise_vid > prob;
      cotags_data(to_flip_off, video_index) = 0;
    end
    data{feature_index} = cotags_data;
  end
end

data = cat(1, data{:});
data = permute(data, [3 4 1 2]);
data = gpuArray(single(data));

if nargout > 0
  labels = double(full(imdb.images.label(:, batch)));
  labels(labels==0) = -1;
  %labels = full(imdb.images.augmented_labels(:, batch)) ;
  % labels has to be W x H x D x N
  labels = permute(labels, [3, 4, 1, 2]);
  varargout{1} = {'input', data, 'label', labels} ;
end