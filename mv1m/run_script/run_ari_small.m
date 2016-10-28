function run_ari_small
addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];
opts.imdbPath = '/mnt/large/pxnguyen/cnn_exp/ari/ari_small_imdb.mat';
opts.expDir = '/mnt/large/pxnguyen/cnn_exp/ari_small';
cnn_mv1m(opts);
