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
opts.train = struct();
opts.train.gpus = gpus;
switch exp_name
  case 'ari_full'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 50000), 5e-6*ones(1, 50000), 5e-7*ones(1, 50000)];
  case 'aria'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 180000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
  case 'aria-cfc'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 180000), 5e-7*ones(1, 80000)];
  case 'aria2'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 150000), 1e-5*ones(1, 150000), 5e-6*ones(1, 150000)];
    opts.only_fc = true;
  case 'ari_full_nospam'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 80000), 5e-6*ones(1, 80000), 5e-7*ones(1, 80000)];
    opts.dropout_ratio = 0.5;
  case 'martin'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 180000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
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
    opts.learning_schedule = [5e-5 * ones(1, 90000), 5e-6*ones(1, 90000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
  case 'ari_mod_vis'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 180000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
  case 'ari_mod_text'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 150000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
  case 'ari_mod_both'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 200000), 5e-6*ones(1, 150000), 5e-7*ones(1, 80000)];
    opts.only_fc = true;
  case 'jaxson'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 20000), 5e-6*ones(1, 40000), 4e-7*ones(1, 40000)];
    opts.only_fc = true;
  case 'danilo_nospam'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 180000), 5e-6*ones(1, 100000), 4e-7*ones(1, 100000)];
  case 'danilo_retrain'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 200000), 5e-6*ones(1, 100000), 4e-7*ones(1, 100000)];
  case 'dani'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 80000), 5e-6*ones(1, 100000), 4e-7*ones(1, 100000)];
    opts.only_fc = true;
  case 'nash'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.learning_schedule = [5e-5 * ones(1, 100000), 5e-6*ones(1, 100000), 4e-7*ones(1, 100000)];
    opts.only_fc = true;
  case 'aria_upperbound'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.num_eval_per_epoch = 4000;
    opts.learning_schedule = [5e-6 * ones(1, 80000), 1e-6*ones(1, 40000), 5e-7*ones(1, 40000)];
    opts.only_fc = true;
    opts.loss_type = 'logistic';
    opts.label_type = 'vetted';clc
  case 'aria_ub2' % sample as aria_upperbound, but sample from 1000 instead of 105.
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.num_eval_per_epoch = 8000;
    opts.learning_schedule = [5e-6 * ones(1, 80000), 1e-6*ones(1, 40000), 5e-7*ones(1, 40000)];
    opts.only_fc = true;
    opts.loss_type = 'logistic';
    opts.label_type = 'vetted';
  case 'aria_softmax_augmented'
    opts.iter_per_epoch = 100000;
    opts.iter_per_save = 2000;
    opts.num_eval_per_epoch = 200;
    opts.learning_schedule = [...
      5e-5*ones(1, 40000), 1e-5*ones(1, 40000), ...
      5e-6*ones(1, 40000), 1e-6*ones(1, 40000),...
      5e-7*ones(1, 40000), 1e-7*ones(1, 40000)];
    opts.only_fc = true;
    opts.label_type = 'original';
    opts.loss_type = 'softmax';
end
rng('shuffle');
cnn_mv1m(opts);