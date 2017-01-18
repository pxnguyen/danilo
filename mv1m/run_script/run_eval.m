function run_eval(exp_name)
run('matconvnet/matlab/vl_setupnn.m');
addpath(genpath('MexConv3D'))
opts = struct();
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/mnt/hermes/nguyenpx/vine-images/';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/home/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
end
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = [1];

opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));

% find the latest trained checkpoint
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d.mat', iter));
opts.model_path = fullfile(opts.expDir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));

opts.train = struct();
opts.train.gpus = [1];
opts.layers_to_store = {'fc1000'};

% extracting the features
if ~exist(opts.resdb_path, 'file')
  cnn_mv1m_evaluate(opts);
end

fprintf('Loading imdb\n');
tic; imdb = load(opts.imdbPath); toc;
train_indeces = find(imdb.images.set==1);
info.train_vid_count = sum(imdb.images.label(:, train_indeces), 2);

fprintf('Loading resdb\n');
tic; resdb = load(opts.resdb_path); toc;

fprintf('Computing APs...\n');
info.AP_tag = compute_average_precision(resdb.fc1000.outputs, resdb.gts);

% compute the prec@k
fprintf('Computing prec@K...\n');
info.prec_at_k = compute_precision_at_k(resdb.fc1000.outputs, resdb.gts);

% computing the adjusted prec@k
if isfield(imdb.images, 'vetted_label')
  fprintf('Computing adjusted prec@K...\n');
  info.adjusted_prec_at_k = compute_precision_at_k(resdb.fc1000.outputs,...
    imdb.images.vetted_label(:, resdb.video_ids));
end

info_path = fullfile(opts.expDir, 'info.mat');
fprintf('saving the AP info to %s\n', info_path);
save(info_path, '-struct', 'info');

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