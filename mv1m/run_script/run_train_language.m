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
opts.batch_size = 9;
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = gpus;
switch exp_name
  case 'aria'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 180000), 5e-7*ones(1, 80000)];
end
rng('shuffle');
cnn_language_train(opts);
