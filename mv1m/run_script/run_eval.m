function run_eval(exp_name)
run('matconvnet/matlab/vl_setupnn.m');
addpath(genpath('MexConv3D'))
opts = struct();
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    opts.expDir = fullfile('mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/tmp/vine-images/'
    opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/home/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end
opts.imdbPath = fullfile(opts.expDir,...
  sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = [1];

[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/mnt/large/pxnguyen/vine-images-test/';
    opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/home/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end

opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));

% find the latest trained checkpoint
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d.mat', iter));
opts.model_path = fullfile(opts.expDir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));

opts.train = struct();
opts.train.gpus = [1];

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

