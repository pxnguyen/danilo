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
opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
opts.num_frame = 10;
opts.batch_size = 9;
opts.layers_to_store = {'pool5', 'sigmoid', 'fc1000'};
opts.set_to_run = 'eval';
opts.input_type = 'video';
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
                                 'pretrained_path', opts.pretrained_path, ...
                                 'batch_size', opts.batch_size, ...
                                 'num_frame', opts.num_frame, ...
                                 'input_type', opts.input_type,...
                                 'classNames', imdb.classes.name);
%       net = cnn_mv1m_init_language_model(...
%         'batch_size', opts.batch_size, ...
%         'classNames', imdb.classes.name);
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
elseif strcmp(opts.set_to_run, 'test')
  val_set = find(imdb.images.set==2);
else
  error('Wrong value for set_to_run %s', opts.set_to_run);
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
    all_files = cat(2, all_files{:});
  end
else
  for video_index = 1:length(video_paths)
    video_name = sprintf('%s.jpg', video_paths{video_index});
    all_files{video_index} = video_name;
  end
end

data = getImageBatch(all_files, opts.(phase), 'prefetch', nargout == 0) ;

% stuffs for the w2v stuffs, need to merge this to the previous block
% data = cell(length(video_paths), 1);
% for video_index = 1:length(video_paths)
%   video_path = video_paths{video_index};
%   [~, name, ~] = fileparts(video_path);
%   w2v_path = fullfile(w2v_storage, sprintf('%s.mat', name));
%   
%   if exist(w2v_path, 'file')
%     % load the mat
%     lstruct = load(w2v_path);
%     if ~isempty(lstruct.all_vecs)
%       data{video_index} = sum(lstruct.all_vecs, 1);
%       success(video_index) = true;
%     end
%   end
% end

% data = data(success);
% data = cat(1, data{:});
% data = permute(data, [3 4 2 1]);
% data = gpuArray(single(data));

% batch = batch(success);

if nargout > 0
  labels = full(imdb.images.label(:, batch)) ;
  labels(labels==0) = -1;
  labels = permute(labels, [3, 4, 1, 2]);
  switch networkType
    case 'simplenn'
      varargout = {data, labels} ;
    case 'dagnn'
      varargout{1} = {'input', data, 'label', labels} ;
  end
end