function run_eval(exp_name)
run('matconvnet/matlab/vl_setupnn.m');
addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];
opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp', exp_name);
opts.imdbPath = fullfile(opts.expDir,...
  sprintf('%s_imdb.mat', exp_name));


% find the latest trained checkpoint
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d.mat', iter));
opts.model_path = fullfile(opts.expDir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));
opts

% extracting the features
if ~exist(opts.resdb_path, 'file')
  cnn_mv1m_evaluate(opts);
end

% compute the mAP
map_opts = struct();
map_opts.resdb_path = opts.resdb_path;
map_opts.imdbPath = opts.imdbPath;
map_opts.expDir = opts.expDir;
cnn_compute_mAP(map_opts);

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

