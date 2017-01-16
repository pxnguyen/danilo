function [net, info] = cnn_mv1m_evaluate(varargin)
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
opts.num_frame = 10;
opts.batch_size = 9;
opts.layers_to_store = {'pool5', 'sigmoid', 'fc1000'};
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
opts
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

cnn_eval_dag(net, imdb, getBatchFn(opts, net.meta), ...
  'expDir', opts.expDir, ...
  'resdb_path', opts.resdb_path,...
  'model_path', opts.model_path,...
  'layers_to_store', opts.layers_to_store,...
  'prefetch', true, ...
  'train', NaN, ...
  'val', find(imdb.images.set==2),...
  opts.train, ...
  net.meta.trainOpts);

% -------------------------------------------------------------------------
function fn = getBatchFn(opts, meta)
% -------------------------------------------------------------------------

useGpu = numel(opts.train.gpus) > 0 ;
bopts.test = struct('useGpu', useGpu);

% Copy the parameters for data augmentation
bopts.train = bopts.test ;
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
data = cell(length(video_paths), 1);
success = false(length(video_paths), 1);
for video_index = 1:length(video_paths)
  video_path = video_paths{video_index};
  [~, name, ~] = fileparts(video_path);
  w2v_path = fullfile(w2v_storage, sprintf('%s.mat', name));
  
  if exist(w2v_path, 'file')
    % load the mat
    lstruct = load(w2v_path);
    if ~isempty(lstruct.all_vecs)
      data{video_index} = sum(lstruct.all_vecs, 1);
      success(video_index) = true;
    end
  end
end

data = data(success);
data = cat(1, data{:});
data = permute(data, [3 4 2 1]);
data = gpuArray(single(data));

batch = batch(success);

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