function run_train(exp_name, gpus)
run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m'))
addpath(genpath('MexConv3D'))
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/mnt/hermes/nguyenpx/vine-images/'
    opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/scratch/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end
opts.num_frame = 10;
opts.batch_size = 9;
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = gpus;
switch exp_name
  case 'ari_full'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 50000), 5e-6*ones(1, 50000), 5e-7*ones(1, 50000)];
  case 'ari_full_nospam'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 80000), 5e-6*ones(1, 80000), 5e-7*ones(1, 80000)];
  case 'ari_half'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 40000), 5e-6*ones(1, 40000), 5e-7*ones(1, 40000)];
  case 'ari_small'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 20000), 5e-6*ones(1, 40000), 4e-7*ones(1, 40000)];
  case 'ari_nospam_small'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 20000), 5e-6*ones(1, 40000), 4e-7*ones(1, 40000)];
  case 'danilo_nospam'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 80000), 5e-6*ones(1, 100000), 4e-7*ones(1, 100000)];
end
rng('shuffle');
cnn_mv1m(opts);
