function run_extract_pool5(model_name, dataset_name)
% model_name: the model name used for extraction
% dataset_name: the dataset where the data is extracted on
% run_extract_pool5('ari_small', 'ari_full');
run('matconvnet/matlab/vl_setupnn.m');
addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];

[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    model_dir = fullfile('/mnt/large/pxnguyen/cnn_exp', model_name);
    dataset_dir = fullfile('/mnt/large/pxnguyen/cnn_exp', dataset_name);
    opts.expDir = model_dir;
    opts.frame_dir = '/mnt/hermes/nguyenpx/vine-images/';
end

opts.imdbPath = fullfile(dataset_dir, sprintf('%s_imdb.mat', dataset_name));

% find the latest trained checkpoint
[epoch, iter] = findLastCheckpoint(model_dir);
opts.resdb_path = fullfile(dataset_dir,...
  sprintf('resdb-model-%s-dataset-%s-iter-%d.mat', model_name, dataset_name, iter));
opts.model_path = fullfile(model_dir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));
opts

% extracting the features
if ~exist(opts.resdb_path, 'file')
  cnn_mv1m_extract_pool5(opts);
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

