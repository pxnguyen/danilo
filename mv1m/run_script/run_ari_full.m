function run_ari_full
run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m'))
addpath(genpath('MexConv3D'))
[~,hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'pi'
    opts.expDir = '/mnt/large/pxnguyen/cnn_exp/ari_full';
    opts.frame_dir = '/tmp/vine-images/'
  case 'omega'
    opts.expDir = '/home/nguyenpx/cnn_exp/ari_full';
    opts.frame_dir = '/scratch/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end
opts.imdbPath = fullfile(opts.expDir, 'ari_full_imdb.mat');
opts.train = struct();
opts.train.gpus = [1];
opts.iter_per_epoch = 20000;
opts.iter_per_save = 1000;
cnn_mv1m(opts);
