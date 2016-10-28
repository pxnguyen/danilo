function run_danilo
addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];
opts.imdbPath = '/mnt/large/pxnguyen/cnn_exp/danilo/imdb.mat';
cnn_mv1m(opts);
