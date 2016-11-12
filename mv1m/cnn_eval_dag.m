function cnn_eval_dag(net, imdb, getBatch, varargin)
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
opts.resdb_path = fullfile(opts.expDir, 'resdb.mat') ;
opts.model_path = fullfile(opts.expDir, 'net-epoch-1-iter-101000.mat');
opts.layers_to_store = {'sigmoid'};
opts = vl_argparse(opts, varargin) ;
opts.batchSize = 16;
opts

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end
if isnan(opts.val), opts.val = [] ; end
prepareGPUs(opts, true) ;

[net, state, stats] = loadState(opts.model_path) ;

val_random_order = 1:numel(opts.val);
batchSize = opts.batchSize;

% Train for one epoch.
params = opts ;
params.epoch = 0;

params.val = opts.val(val_random_order);
params.imdb = imdb ;
params.getBatch = getBatch ;

for layer_index = 1:length(params.layers_to_store)
  layer_name = opts.layers_to_store{layer_index};
  sel = find(cellfun(@(x) strcmp(x, layer_name), {net.vars.name}));
  net.vars(sel).precious = 1;
end

processEpoch(net, state, params, 'val') ;

fprintf('Combining all the resdb\n');
[~, name, ~] = fileparts(params.resdb_path);
list = dir(fullfile(opts.expDir, sprintf('%s-part-*.mat', name)));
resdb = struct();
resdb.names = cell(length(list), 1);
resdb.predictions = cell(length(list), 1);
for list_index = 1:length(list)
  fprintf('%d/%d\n', list_index, length(list));
  part_resdb = load(fullfile(opts.expDir, list(list_index).name));
  resdb.names{list_index} = part_resdb.name;
  resdb.video_ids{list_index} = part_resdb.video_ids;
  resdb.gts{list_index} = part_resdb.gts;
  for layer_index = 1:numel(opts.layers_to_store)
    layer_name = opts.layers_to_store{layer_index};
    resdb.(layer_name).outputs{list_index} =...
      part_resdb.(layer_name).outputs;
  end
  clear part_resdb
end

resdb.names = cat(2, resdb.names{:});
resdb.video_ids = cat(2, resdb.video_ids{:});
for layer_index = 1:numel(opts.layers_to_store)
  layer_name = opts.layers_to_store{layer_index};
  resdb.(layer_name).outputs =...
    cat(2, resdb.(layer_name).outputs{:});
end

% combine all the saved features
save(opts.resdb_path, '-v7.3', '-struct', 'resdb');


% -------------------------------------------------------------------------
function resdb = processEpoch(net, state, params, mode)
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

num = 0 ;
epoch = params.epoch ;
subset = params.(mode) ;
adjustTime = 0 ;

stats.num = 0 ; % return something even if subset = []
stats.time = 0 ;

start = tic ;
max_iter = ceil(numel(subset)/params.batchSize);
batch_index = findLastResultDB(params.resdb_path);

save_iter = 2000;
bs = params.batchSize;
local_batch_index = 1;
max_batch = ceil(numel(subset)/params.batchSize);
resdb = struct();

for t=1+batch_index*bs:bs:numel(subset)
  fprintf('%s: epoch %02d: %3d/%3d:', mode, epoch, ...
          fix((t-1)/params.batchSize)+1, ceil(numel(subset)/params.batchSize)) ;
  batchSize = min(params.batchSize, numel(subset) - t + 1) ;

  for s=1:params.numSubBatches
    % get this image batch and prefetch the next
    batchStart = t + (labindex-1) + (s-1) * numlabs ;
    batchEnd = min(t+params.batchSize-1, numel(subset)) ;
    batch = subset(batchStart : params.numSubBatches * numlabs : batchEnd) ;
    num = num + numel(batch) ;
    if numel(batch) == 0, continue ; end

    inputs = params.getBatch(params.imdb, batch) ;

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

    net.mode = 'normal' ;
    net.eval(inputs) ; % forward pass

    % find the sigmoid layers
    resdb.name{local_batch_index} = params.imdb.images.name(batch);
    resdb.video_ids{local_batch_index} = batch;
    resdb.gts{local_batch_index} = permute(inputs{4}, [3 4 1 2]);
    for layer_index = 1:length(params.layers_to_store)
      layer_name = params.layers_to_store{layer_index};
      sel = find(cellfun(@(x) strcmp(x, layer_name), {net.vars.name})) ;
      resdb.(layer_name).outputs{local_batch_index} =...
        gather(permute(net.vars(sel).value, [3 4 1 2]));
    end

    local_batch_index = local_batch_index + 1;
  end

  batch_index = batch_index + 1;

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
  fprintf('\n') ;

  cur_batch = fix((t-1)/params.batchSize)+1;
  if mod(batch_index, save_iter) == 0 || cur_batch == max_iter
    fprintf('Saving the intermediate resdb\n');
    resdb.name = cat(2, resdb.name{:});
    resdb.video_ids = cat(2, resdb.video_ids{:});
    resdb.gts = cat(2, resdb.gts{:});
    for layer_index = 1:length(params.layers_to_store)
      layer_name = params.layers_to_store{layer_index};
      resdb.(layer_name).outputs = cat(2, resdb.(layer_name).outputs{:});
    end
    [path, name, ext] = fileparts(params.resdb_path);
    resdb_temp_filepath = fullfile(path,...
      [name sprintf('-part-%05d', batch_index) ext]);
    save(resdb_temp_filepath, '-struct', 'resdb');

    % reset the counter
    local_batch_index = 1;
    resdb = struct();
  end
end

% Save back to state.
state.stats.(mode) = stats ;
if ~params.saveMomentum
  state.momentum = [] ;
else
  state.momentum = cellfun(@gather, state.momentum, 'uniformoutput', false) ;
end

net.reset() ;
net.move('cpu') ;

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
function batch_index = findLastResultDB(resdb_path)
% -------------------------------------------------------------------------
[modelDir, name, ~] = fileparts(resdb_path);
pat = sprintf('%s-part-*.mat', name);
list = dir(fullfile(modelDir, pat)) ;
pat2 = [name '-part-([\d]+).mat'];
tokens = regexp({list.name}, pat2, 'tokens') ;
batch_index = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
batch_index = max([batch_index 0]);

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
if numGpus >= 1 && cold
  fprintf('%s: resetting GPU\n', mfilename)
  clearMex() ;
  if numGpus == 1
    gpuDevice(opts.gpus)
  else
    spmd
      clearMex() ;
      gpuDevice(opts.gpus(labindex))
    end
  end
end
