addpath(genpath('MexConv3D'))
opts = struct();
opts.train = struct();
opts.train.gpus = [1];
cnn_mv1m(opts);
