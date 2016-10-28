function run_ari
addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];
opts.imdbPath = '/mnt/large/pxnguyen/cnn_exp/ari/ari_imdb.mat';
opts.expDir = '/mnt/large/pxnguyen/cnn_exp/ari';
cnn_mv1m(opts);
