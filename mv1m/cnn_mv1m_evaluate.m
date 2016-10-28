function [net, info] = cnn_mv1m_evaluate(varargin)
%CNN_IMAGENET   Demonstrates training a CNN on ImageNet
run(fullfile('/home/phuc/Research/danilo/mv1m/matconvnet', 'matlab', 'vl_setupnn.m'));

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

opts.numFetchThreads = 8 ;
opts.lite = false ;
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts.resdb_path = fullfile(opts.expDir, 'resdb.mat');
opts.model_path = fullfile(opts.expDir, 'net-epoch-1-iter-101000.mat');
opts.train = struct() ;
opts = vl_argparse(opts, varargin) ;
if ~isfield(opts.train, 'gpus'), opts.train.gpus = []; end;

% -------------------------------------------------------------------------
%                                                              Prepare data
% -------------------------------------------------------------------------

imdb = load(opts.imdbPath) ;
imdb.imageDir = fullfile(opts.dataDir, 'videos');

% Compute image statistics (mean, RGB covariances, etc.)
imageStatsPath = fullfile(opts.expDir, 'imageStats.mat') ;
if exist(imageStatsPath)
  load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
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
  'prefetch', true, ...
  opts.train, ...
  net.meta.trainOpts);

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
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if isempty(images)
  return
end
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end
num_frames = 5;
video_paths = images;
all_files = cell(length(video_paths),1);
for video_index = 1:length(video_paths)
  files = extract_frames(video_paths{video_index},...
    'dest_dir', '/mnt/large/pxnguyen/vine-images-test');
  all_files{video_index} = files(1:num_frames);
end
all_files = cat(2, all_files{:});
data = getImageBatch(all_files, opts.(phase), 'prefetch', nargout == 0) ;
if nargout > 0
  labels = full(imdb.images.label(:, batch)) ;
  labels(labels==0) = -1;
  num_classes = numel(imdb.classes.name);
  % labels has to be W x H x D x N
  labels = permute(labels, [3, 4, 1, 2]);
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end
