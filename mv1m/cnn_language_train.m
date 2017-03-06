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

if exist(opts.imdbPath, 'file')
  imdb = load(opts.imdbPath) ;
  imdb.imageDir = fullfile(opts.dataDir, 'videos');
else
  error('imdb not found at %s', opts.imdbPath);
end

if exist(fullfile(opts.expDir, 'latent_labels.mat'), 'file')
  fprintf('Loading latent labels...\n');
  latent_labels = load(fullfile(opts.expDir, 'latent_labels.mat'));
  imdb.latent_labels = latent_labels;
end

if exist(fullfile(opts.expDir, 'fc1000.mat'), 'file')
  fprintf('Loading predictions...\n');
  lstruct = load(fullfile(opts.expDir, 'fc1000.mat'));
  imdb.images.fc1000 = lstruct.fc1000;
end

% -------------------------------------------------------------------------
%                                                             Prepare model
% -------------------------------------------------------------------------

pretrained_path = fullfile(opts.expDir, 'net-epoch-0-iter-0.mat');

if isempty(opts.network)
  switch opts.modelType
    case 'resnet-50'
      net = cnn_mv1m_init_language_model(...
        'learning_schedule', opts.learning_schedule, ...
        'batch_size', opts.batch_size, ...
        'features', opts.features, ...
        'pretrained_path', pretrained_path, ...
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
  'derOutputs', {'loss1', 1},...
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
switch opts.features{1}
  case 'rescore'
    fn = @(x, y) get_batch_rescore(bopts, useGpu, lower(opts.networkType), x, y);
  case 'cotags'
    fn = @(x, y) getBatch(bopts, useGpu, lower(opts.networkType), x, y);
  otherwise
    error('Unrecognized feature %s', opts.features{1});
end

% -------------------------------------------------------------------------
function varargout = getBatch(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if isempty(images); return; end;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end

video_paths = images;

augmented_labels = imdb.images.label(:, batch);
data = cell(numel(opts.features), 1);
for feature_index = 1:numel(opts.features)
  feature = opts.features{feature_index};
  if strcmp(feature, 'desc')
    error('This code is not working right now. Needs to be fixed');
    w2v_storage = '/home/phuc/Research/word2vec_storage';
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
    observed_input = full(imdb.images.label(:, batch));
    corrupted_input = full(imdb.images.label(:, batch));
    rand_noise = rand(size(corrupted_input));
    prob = 0.5;
    for video_index = 1:length(video_paths)
      rand_noise_vid = rand_noise(:, video_index);
      to_flip_off = rand_noise_vid > prob;
      corrupted_input(to_flip_off, video_index) = 0;
    end
    corrupted_input = permute(corrupted_input, [3 4 1 2]);
    observed_input = permute(observed_input, [3 4 1 2]);
    corrupted_input = gpuArray(single(corrupted_input));
    observed_input = gpuArray(single(observed_input));
    
    latent_label = zeros(4000, numel(batch), 5);
    neighbor_idx = imdb.closest_neighbors(:, batch);
    for i=1:5
      latent_label(imdb.tags_to_train, :, i) = ...
        imdb.latent_labels.soft_labels(:, neighbor_idx(i, :));
    end
    latent_label = max(latent_label, [], 3);
    latent_label = permute(latent_label, [3 4 1 2]);
    latent_label = gpuArray(single(latent_label));
    
    % loading the latent labels
    if strcmp(phase, 'train')
      varargout{1} = {'corrupted_input', corrupted_input,...
        'observed_input', observed_input,...
        'latent_label', latent_label};
    else
      varargout{1} = {'corrupted_input', corrupted_input,...
        'observed_input', observed_input,...
        'latent_label', observed_input};
    end
  end
end

% -------------------------------------------------------------------------
function varargout = get_batch_rescore(opts, useGpu, networkType, imdb, batch)
% -------------------------------------------------------------------------
images = strcat([imdb.imageDir filesep], imdb.images.name(batch)) ;
if isempty(images); return; end;
if ~isempty(batch) && imdb.images.set(batch(1)) == 1
  phase = 'train' ;
else
  phase = 'test' ;
end

for feature_index = 1:numel(opts.features)
  feature = opts.features{feature_index};
  if strcmp(feature, 'rescore')
    observed_labels = full(imdb.images.label(imdb.tags_to_train, batch));
    preds = full(vl_nnsigmoid(imdb.images.fc1000(:, batch)));
    inputs = cat(1, observed_labels, preds);
    inputs = permute(inputs, [3 4 1 2]);
    inputs = gpuArray(single(inputs));
    
    combined_labels = full(imdb.images.combined_label(imdb.tags_to_train, batch));
    
    combined_labels = permute(combined_labels, [3 4 1 2]);
    combined_labels = gpuArray(single(combined_labels));
    
    % loading the latent labels
    if strcmp(phase, 'train')
      varargout{1} = {'input', inputs,...
        'labels', combined_labels};
    else
      varargout{1} = {'input', inputs,...
        'labels', combined_labels};
    end
  end
end

% if nargout > 0
%   labels = double(full(imdb.images.label(:, batch)));
%   labels(labels==0) = -1;
%   %labels = full(imdb.images.augmented_labels(:, batch)) ;
%   % labels has to be W x H x D x N
%   labels = permute(labels, [3, 4, 1, 2]);
%   varargout{1} = {'input', data, 'label', labels} ;
% end