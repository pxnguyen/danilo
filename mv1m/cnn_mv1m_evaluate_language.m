function [net, info] = cnn_mv1m_evaluate_language(varargin)
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
opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/danilo');
opts.frame_dir = '';
[opts, varargin] = vl_argparse(opts, varargin) ;

opts.numFetchThreads = 8 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.resdb_path = fullfile(opts.expDir, 'resdb.mat');
opts.model_path = fullfile(opts.expDir, 'net-epoch-1-iter-101000.mat');
opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
opts.batch_size = 9;
opts.layers_to_store = {'fc1000'};
opts.features = {'desc', 'cotags'};
opts.train = struct() ;
opts.set_to_run = 'eval';
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

imdb = load(opts.imdbPath) ;
imdb.imageDir = fullfile(opts.dataDir, 'videos');

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_mv1m_init_language_model(...
        'batch_size', opts.batch_size, ...
        'features', opts.features,...
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

if strcmp(opts.set_to_run, 'train')
  val_set = find(imdb.images.set==1);
else
  val_set = find(imdb.images.set==2);
end

cnn_eval_dag(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  'resdb_path', opts.resdb_path,...
  'model_path', opts.model_path,...
  'layers_to_store', opts.layers_to_store,...
  'prefetch', true, ...
  'train', NaN, ...
  'val', val_set,...
  opts.train, ...
  net.meta.trainOpts);

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
if isempty(images)
  return
end
w2v_storage = '/home/phuc/Research/word2vec_storage';
video_paths = images;

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
    cotags_data = full(imdb.images.label(:, batch)) ;
    data{feature_index} = cotags_data;
  end
end

data = cat(1, data{:});
data = permute(data, [3 4 1 2]);
data = gpuArray(single(data));

if nargout > 0
  labels = full(imdb.images.label(:, batch)) ; % tem fix
  labels(labels==0) = -1;
  % labels has to be W x H x D x N
  labels = permute(labels, [3, 4, 1, 2]);
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end