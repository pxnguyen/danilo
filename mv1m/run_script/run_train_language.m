function run_train_language(exp_name, gpus)
run(fullfile('matconvnet', 'matlab', 'vl_setupnn.m'))
addpath(genpath('MexConv3D'))
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
  case 'epsilon'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.dataDir = '/home/nguyenpx/vine-large-2';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.dataDir = '/home/nguyenpx/vine-large-2';
end
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = gpus;
switch exp_name
  case 'aria_language'
    opts.batch_size = 40;
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [1e-1 * ones(1, 40000), 1e-2*ones(1, 30000), 1e-3*ones(1, 20000)];
  case 'aria_desc'
    opts.batch_size = 40;
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [1e-1 * ones(1, 40000), 1e-2*ones(1, 30000), 1e-3*ones(1, 20000)];
    opts.features = {'desc'};
  case 'aria_desc_cotags'
    opts.batch_size = 40;
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [1e-1 * ones(1, 30000), 1e-2*ones(1, 20000), 1e-3*ones(1, 10000), 1e-4*ones(1, 10000)];
    opts.features = {'desc', 'cotags'};
  case 'aria_cotags'
    opts.batch_size = 40;
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [1e-1 * ones(1, 40000), 1e-2*ones(1, 30000), 1e-3*ones(1, 20000)];
    opts.features = {'cotags'};
end
rng('shuffle');
cnn_language_train(opts);
