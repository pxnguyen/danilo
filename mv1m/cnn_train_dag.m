function [net,stats] = cnn_train_dag(net, imdb, getBatch, varargin)
%CNN_TRAIN_DAG Demonstrates training a CNN using the DagNN wrapper
%    CNN_TRAIN_DAG() is similar to CNN_TRAIN(), but works with
%    the DagNN wrapper instead of the SimpleNN wrapper.

% Copyright (C) 2014-16 Andrea Vedaldi.
% All rights reserved.
%
% This file is part of the VLFeat library and is made available under
% the terms of the BSD license (see the COPYING file).

opts.expDir = fullfile('data','exp') ;
opts.continue = true ;
opts.batchSize = 256 ;
opts.numSubBatches = 1 ;
opts.train = [] ;
opts.val = [] ;
opts.gpus = [] ;
opts.prefetch = false ;
opts.numEpochs = 300 ;
opts.learningRate = 0.001 ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.saveMomentum = true ;
opts.nesterovUpdate = false ;
opts.randomSeed = 0 ;
opts.profile = false ;
opts.parameterServer.method = 'mmap' ;
opts.parameterServer.prefix = 'mcn' ;

opts.derOutputs = {'objective', 1} ;
opts.extractStatsFn = @extractStats ;
opts.plotStatistics = true;
opts.iter_per_epoch = 80000;
opts.iter_per_save = 1000;
opts.num_eval_per_epoch = 8000;
opts.label_type = 'original';
opts.loss_type = 'logistic';
opts = vl_argparse(opts, varargin) ;
opts

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end

% -------------------------------------------------------------------------
%                                                            Initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;
if ~evaluateMode
  if isempty(opts.derOutputs)
    error('DEROUTPUTS must be specified when training.\n') ;
  end
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

modelPath = @(ep, iter) fullfile(opts.expDir, sprintf('net-epoch-%d-iter-%d.mat', ep, iter));
modelFigPath = fullfile(opts.expDir, 'net-train.pdf') ;

[start_epoch, start_iter] = findLastCheckpoint(opts.expDir) ;
if start_epoch >= 1 && start_iter >= 1
  fprintf('%s: resuming by loading epoch %d iter %d\n', mfilename, start_epoch, start_iter) ;
  [net, state, stats] = loadState(modelPath(start_epoch, start_iter)) ;
  save_index = length(stats.iter_recorded);
elseif start_epoch == 0 && start_iter == 0
  [net, ~, ~] = loadState(modelPath(start_epoch, start_iter)) ;
  save_index = 1;
  start_epoch = 1;
  state = [] ;
else
  save_index = 1;
  state = [] ;
end

sel = find(cellfun(@(x) strcmp(x, 'fc1000'), {net.vars.name}));
net.vars(sel).precious = 1;

epoch = start_epoch;
current_iter = start_iter;
val_with_relevant_labels = sum(imdb.images.label(:, opts.val), 1) > 0;
opts.val = opts.val(val_with_relevant_labels);
val_random_order = randperm(numel(opts.val));
batchSize = opts.batchSize;
iter_per_epoch = opts.iter_per_epoch;
iter_per_save = opts.iter_per_save;
max_iter = ceil(numel(opts.train)/opts.batchSize);
prepareGPUs(opts, true) ;
done = false;
while ~done
  % Set the random seed based on the epoch and opts.randomSeed.
  % This is important for reproducibility, including when training
  % is restarted from a checkpoint.
  if (current_iter - start_iter) >= iter_per_epoch
    % every N number iter
    % shuffle the training data again by increasing the epoch
    % move the last epoch changed
    start_iter = current_iter;
    epoch = epoch + 1;
    val_random_order = randperm(numel(opts.val));
    prepareGPUs(opts, true) ;
  end

  % Train for one epoch.
  params = opts ;
  params.epoch = epoch ;
  params.learningRate = opts.learningRate(min(current_iter+1, numel(opts.learningRate))) ;

  %TODO(phuc): need to do the class balancing here
  fprintf('Shuffling and balancing the data...\n');
  [train_order, imdb, augmented_labels] = select_training_examples(...
    iter_per_save*batchSize, imdb,...
    'is_train', true, 'label_type', opts.label_type, 'loss_type', opts.loss_type);
%   train_order = randperm(numel(opts.train));
%   train_order = train_order(1:min(iter_per_save*batchSize,...
%     numel(train_order)));
%   imdb.images.augmented_labels = imdb.images.label;
  %params.train = opts.train(train_random_order) ; % shuffle
  params.train = struct();
  params.train.order = train_order ; % shuffle
  params.train.augmented_labels = augmented_labels ; % shuffle
  
  [val_order, imdb, augmented_labels_eval] = select_training_examples(...
    opts.num_eval_per_epoch, imdb,...
    'is_train', false);
  
  params.val = struct();
  params.val.order = val_order;
  params.val.augmented_labels = augmented_labels_eval;
%   params.val = opts.val(val_random_order(1:min(opts.num_eval_per_epoch, numel(val_random_order))));
  
  params.imdb = imdb ;
  params.getBatch = getBatch ;
  params.current_iter = current_iter;

  if numel(opts.gpus) <= 1
    [net, state] = processEpoch(net, state, params, 'train') ;
    current_iter = current_iter + iter_per_save;
    [net, state] = processEpoch(net, state, params, 'val') ;
    if ~evaluateMode
      removeSmallestCheckpoint(opts.expDir);
      saveState(modelPath(epoch, current_iter), net, state) ;
    end
    lastStats = state.stats ;
  else
    spmd
      [net, state] = processEpoch(net, state, params, 'train') ;
      [net, state] = processEpoch(net, state, params, 'val') ;
      if labindex == 1 && ~evaluateMode
        saveState(modelPath(epoch), net, state) ;
      end
      lastStats = state.stats ;
    end
    lastStats = accumulateStats(lastStats) ;
  end

  stats.train(save_index) = lastStats.train ;
  stats.val(save_index) = lastStats.val ;
  stats.iter_recorded(save_index) = current_iter;
  stats.learning_rate(save_index) = params.learningRate;
  save_index = save_index + 1;
  clear lastStats ;
  saveStats(modelPath(epoch, current_iter), stats) ;

  fprintf('current_epoch: %d, current_iter: %d, , save_index: %d\n',...
    epoch,...
    current_iter,...
    save_index);
  iter_to_change = iter_per_epoch - (current_iter - start_iter);
  fprintf('iter to epoch change + shuffling: %d\n', iter_to_change);

  if opts.plotStatistics
    switchFigure(1) ; clf ;
    plots = setdiff(...
      cat(2,...
      fieldnames(stats.train)', ...
      fieldnames(stats.val)'), {'num', 'time'}) ;
    num_plots = numel(plots);
    for f = {'train', 'val'}
      f = char(f) ;
      for p = plots
        if strcmp(f, 'train')
          subplot(numel(plots), 2, find(strcmp(p,plots))) ;
          fmt = '-o';
        else
          subplot(numel(plots), 2, find(strcmp(p,plots))+num_plots) ;
          fmt = 'r-o';
        end
        if strcmp(p, 'APs')
          vals = {stats.(f).APs};
          mean_val = cellfun(@(x) mean(NaNproof(x, imdb.tags_to_train)), vals);
          std_val = cellfun(@(x) std(NaNproof(x, imdb.tags_to_train)), vals);
          errorbar(stats.iter_recorded, mean_val, std_val, fmt);
        elseif strcmp(p, 'prec_at_k')
          vals = {stats.(f).prec_at_k};
          mean_val = cellfun(@(x) mean(NaNproof(x, imdb.tags_to_train)), vals);
          std_val = cellfun(@(x) std(NaNproof(x, imdb.tags_to_train)), vals);
          errorbar(stats.iter_recorded, mean_val, std_val, fmt);
        else
          values = zeros(0, length(stats.iter_recorded)) ;
          p = char(p) ;
          leg = {} ;
          if isfield(stats.(f), p)
            tmp = [stats.(f).(p)] ;
            values(end+1,:) = tmp(1,:)' ;
            leg{end+1} = f ;
          end
          plot(stats.iter_recorded, values', fmt) ;
        end
        xlabel('iterations') ;
        title(p) ;
        grid on ;
      end
    end
    drawnow ;

    %TODO(phucng): remove this by fixing the print functionality on pi
    [~, hostname] = system('hostname');
    hostname = strtrim(hostname);
    if ~strcmp(hostname, 'pi')
      print(1, modelFigPath, '-dpdf') ;
    end
  end
end

% With multiple GPUs, return one copy
if isa(net, 'Composite'), net = net{1} ; end

% -------------------------------------------------------------------------
function APs = NaNproof(APs, filter)
APs = APs(filter);
APs = APs(~isnan(APs));
% -------------------------------------------------------------------------

% -------------------------------------------------------------------------
function [net, state] = processEpoch(net, state, params, mode)
% -------------------------------------------------------------------------
% Args:
%   net: the network structure
%   state: the information (loss + other performance)
%   params: structure containing the running settings
%   mode: specify whether this is train or test
% Note that net is not strictly needed as an output argument as net
% is a handle class. However, this fixes some aliasing issue in the
% spmd caller.

% initialize with momentum 0
if isempty(state) || isempty(state.momentum)
  state.momentum = num2cell(zeros(1, numel(net.params))) ;
end

% move CNN  to GPU as needed
numGpus = numel(params.gpus) ;
if numGpus >= 1
  net.move('gpu') ;
  state.momentum = cellfun(@gpuArray, state.momentum, 'uniformoutput', false) ;
end
if numGpus > 1
  parserv = ParameterServer(params.parameterServer) ;
  net.setParameterServer(parserv) ;
else
  parserv = [] ;
end

% profile
if params.profile
  if numGpus <= 1
    profile clear ;
    profile on ;
  else
    mpiprofile reset ;
    mpiprofile on ;

  end
end

num = 0 ;
epoch = params.epoch ;
subset = params.(mode).order ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

start = tic ;

resdb = struct();
local_batch_index = 1;
for t=1:params.batchSize:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    batch_labels = params.(mode).augmented_labels(...
      batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = params.getBatch(params.imdb, batch) ;
    % get the original label
    original_labels = inputs{4};
    
    % if softmax, use the augmented labels instead
    if strcmp(params.loss_type, 'softmax')
      inputs{4} = batch_labels;
    end

    if params.prefetch
      if s == params.numSubBatches
        batchStart = t + (labindex-1) + params.batchSize ;
        batchEnd = min(t+2*params.batchSize-1, numel(subset)) ;
      else
        batchStart = batchStart + numlabs ;
      end
      nextBatch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
      params.getBatch(params.imdb, nextBatch) ;
    end

    net.meta.curBatchSize = numel(batch);

    if strcmp(mode, 'train')
      net.mode = 'normal' ;
      net.accumulateParamDers = (s ~= 1) ;
      net.eval(inputs, params.derOutputs, 'holdOn', s < params.numSubBatches) ;
    else
      net.mode = 'eval' ;
      net.eval(inputs) ;
    end

    if strcmp(params.loss_type, 'softmax')
      resdb.gts{local_batch_index} = original_labels;
    else
      resdb.gts{local_batch_index} = permute(inputs{4}, [3 4 1 2]);
    end
    sel = find(cellfun(@(x) strcmp(x, 'fc1000'), {net.vars.name})) ;
    resdb.fc1000.outputs{local_batch_index} =...
      gather(permute(net.vars(sel).value, [3 4 1 2]));
    local_batch_index = local_batch_index  + 1;
  end

  % Accumulate gradient.
  if strcmp(mode, 'train')
    if ~isempty(parserv), parserv.sync() ; end
    state = accumulateGradients(net, state, params, batchSize, parserv) ;
  end

  % Get statistics.
  time = toc(start) + adjustTime ;
  batchTime = time - stats.time ;
  stats.num = num ;
  stats.time = time ;
  stats = params.extractStatsFn(stats,net) ;
  currentSpeed = batchSize / batchTime ;
  averageSpeed = (t + batchSize - 1) / time ;
  if t == 3*params.batchSize + 1
    % compensate for the first three iterations, which are outliers
    adjustTime = 4*batchTime - time ;
    stats.time = time + adjustTime ;
  end

  fprintf(' %.1f (%.1f) Hz', averageSpeed, currentSpeed) ;
  for f = setdiff(fieldnames(stats)', {'num', 'time'})
    f = char(f) ;
    fprintf(' %s: %.3f', f, stats.(f)) ;
  end
  fprintf('\n');
end

% resdb.gts = cat(2, resdb.gts{:});
% resdb.gts(resdb.gts==-1) = 0;
% resdb.fc1000.outputs = cat(2, resdb.fc1000.outputs{:});
% fprintf('Computing evaluation metrics\n');
% stats.APs = compute_average_precision(resdb.fc1000.outputs, resdb.gts);
% stats.prec_at_k = compute_precision_at_k(resdb.fc1000.outputs, resdb.gts,...
%   'k', 10);

% Save back to state.
state.stats.(mode) = stats ;
if params.profile
  if numGpus <= 1
    state.prof.(mode) = profile('info') ;
    profile off ;
  else
    state.prof.(mode) = mpiprofile('info');
    mpiprofile off ;
  end
end
if ~params.saveMomentum
  state.momentum = [] ;
else
  state.momentum = cellfun(@gather, state.momentum, 'uniformoutput', false) ;
end

net.reset() ;
net.move('cpu') ;

% -------------------------------------------------------------------------
function state = accumulateGradients(net, state, params, batchSize, parserv)
% -------------------------------------------------------------------------
numGpus = numel(params.gpus) ;
otherGpus = setdiff(1:numGpus, labindex) ;

for p=1:numel(net.params)

  if ~isempty(parserv)
    parDer = parserv.pullWithIndex(p) ;
  else
    parDer = net.params(p).der ;
  end

  switch net.params(p).trainMethod

    case 'average' % mainly for batch normalization
      thisLR = net.params(p).learningRate ;
      if thisLR>0
        net.params(p).value = vl_taccum(...
            1 - thisLR, net.params(p).value, ...
            (thisLR/batchSize/net.params(p).fanout),  parDer) ;
      end

    case 'gradient'
      thisDecay = params.weightDecay * net.params(p).weightDecay ;
      thisLR = params.learningRate * net.params(p).learningRate ;
      
      if thisLR>0 || thisDecay>0
        % Normalize gradient and incorporate weight decay.
        parDer = vl_taccum(1/batchSize, parDer, ...
                           thisDecay, net.params(p).value) ;

        % Update momentum.
        state.momentum{p} = vl_taccum(...
          params.momentum, state.momentum{p}, ...
          -1, parDer) ;

        % Nesterov update (aka one step ahead).
        if params.nesterovUpdate
          delta = vl_taccum(...
            params.momentum, state.momentum{p}, ...
            -1, parDer) ;
        else
          delta = state.momentum{p} ;
        end

        % Update parameters.
        net.params(p).value = vl_taccum(...
          1,  net.params(p).value, thisLR, delta) ;
      end

    otherwise
      error('Unknown training method ''%s'' for parameter ''%s''.', ...
        net.params(p).trainMethod, ...
        net.params(p).name) ;
  end
end

% -------------------------------------------------------------------------
function stats = accumulateStats(stats_)
% -------------------------------------------------------------------------

for s = {'train', 'val'}
  s = char(s) ;
  total = 0 ;

  % initialize stats stucture with same fields and same order as
  % stats_{1}
  stats__ = stats_{1} ;
  names = fieldnames(stats__.(s))' ;
  values = zeros(1, numel(names)) ;
  fields = cat(1, names, num2cell(values)) ;
  stats.(s) = struct(fields{:}) ;

  for g = 1:numel(stats_)
    stats__ = stats_{g} ;
    num__ = stats__.(s).num ;
    total = total + num__ ;

    for f = setdiff(fieldnames(stats__.(s))', 'num')
      f = char(f) ;
      stats.(s).(f) = stats.(s).(f) + stats__.(s).(f) * num__ ;

      if g == numel(stats_)
        stats.(s).(f) = stats.(s).(f) / total ;
      end
    end
  end
  stats.(s).num = total ;
end

% -------------------------------------------------------------------------
function stats = extractStats(stats, net)
% -------------------------------------------------------------------------
sel = find(cellfun(@(x) isa(x,'dagnn.Loss'), {net.layers.block})) ;
for i = 1:numel(sel)
  stats.(net.layers(sel(i)).outputs{1}) = net.layers(sel(i)).block.average ;
end

% -------------------------------------------------------------------------
function saveState(fileName, net_, state)
% -------------------------------------------------------------------------
net = net_.saveobj() ;
save(fileName, 'net', 'state') ;

% -------------------------------------------------------------------------
function saveStats(fileName, stats)
% -------------------------------------------------------------------------
if exist(fileName)
  save(fileName, 'stats', '-append') ;
else
  save(fileName, 'stats') ;
end

% -------------------------------------------------------------------------
function [net, state, stats] = loadState(fileName)
% -------------------------------------------------------------------------
load(fileName, 'net', 'state', 'stats') ;
net = dagnn.DagNN.loadobj(net) ;
if isempty(whos('stats'))
  error('Epoch ''%s'' was only partially saved. Delete this file and try again.', ...
        fileName) ;
end

% -------------------------------------------------------------------------
function [epoch, iter] = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*-iter-*.mat')) ;
if ~isempty(list)
  tokens = regexp({list.name}, 'net-epoch-([\d]+)-iter-([\d]+).mat', 'tokens') ;
  epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
  iter = cellfun(@(x) sscanf(x{1}{2}, '%d'), tokens) ;
  epoch = max([epoch 0]);
  iter = max([iter 0]);
else
  epoch = 1;
  iter = 0;
end

% -------------------------------------------------------------------------
function removeSmallestCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*-iter-*.mat')) ;
[~, idx] = sort([list.datenum]);
if length(list) > 10 % remove the smallest
  todelete = fullfile(modelDir, list(idx(1)).name);
  delete(todelete);
end

% -------------------------------------------------------------------------
function switchFigure(n)
% -------------------------------------------------------------------------
if get(0,'CurrentFigure') ~= n
  try
    set(0,'CurrentFigure',n) ;
  catch
    figure(n) ;
  end
end

% -------------------------------------------------------------------------
function clearMex()
% -------------------------------------------------------------------------
clear vl_tmove vl_imreadjpeg ;

% -------------------------------------------------------------------------
function prepareGPUs(opts, cold)
% -------------------------------------------------------------------------
numGpus = numel(opts.gpus) ;
if numGpus > 1
  % check parallel pool integrity as it could have timed out
  pool = gcp('nocreate') ;
  if ~isempty(pool) && pool.NumWorkers ~= numGpus
    delete(pool) ;
  end
  pool = gcp('nocreate') ;
  if isempty(pool)
    parpool('local', numGpus) ;
    cold = true ;
  end

end
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus(1))
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end