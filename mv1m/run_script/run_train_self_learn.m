function run_train_self_learn(exp_name, gpus)
run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m'))
addpath(genpath('MexConv3D'))
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/mnt/hermes/nguyenpx/vine-images/';
    opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'epsilon'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/home/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/scratch/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end
opts.num_frame = 10;
opts.batch_size = 9;

opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d.mat', iter));
opts.train = struct();
opts.train.gpus = gpus;
switch exp_name
  case 'aria_self_learn'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 50000), 5e-6*ones(1, 50000), 5e-7*ones(1, 50000)];
    opts.model_path = fullfile(opts.expDir, 'net-epoch-7-iter-646000.mat');
    opts.resdb_path = fullfile(opts.expDir, 'resdb.mat');
    opts.imdbPath = fullfile(opts.expDir, 'aria_imdb.mat');
end
rng('shuffle');
cnn_mv1m_self_learn(opts);

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