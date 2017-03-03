function run_eval(exp_name, mode)
run('matconvnet/matlab/vl_setupnn.m');
addpath(genpath('MexConv3D'))
opts = struct();
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
    opts.frame_dir = '/mnt/hermes/nguyenpx/vine-images/';
    opts.pretrained_path = '/home/phuc/Research/pretrained_models/imagenet-resnet-50-dag.mat';
  case 'omega'
    opts.expDir = fullfile('/home/nguyenpx/cnn_exp/', exp_name);
    opts.frame_dir = '/home/nguyenpx/vine-images/';
    opts.dataDir = '/home/nguyenpx/vine-large-2';
    opts.pretrained_path = '/home/nguyenpx/pretrained_models/imagenet-resnet-50-dag.mat';
end
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.train = struct();
opts.train.gpus = [1];

opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));

% find the latest trained checkpoint
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d-%s.mat', iter, mode));
opts.model_path = fullfile(opts.expDir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));

opts.train = struct();
opts.train.gpus = [1];
opts.set_to_run = mode;
switch exp_name
  case 'aria128'
    opts.layers_to_store = {'fc128', 'fc1000'};
  case 'aria'
    opts.layers_to_store = {'pool5', 'fc1000'};
  otherwise
end

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
prob = resdb.fc1000.outputs(imdb.selected, :);
gts = full(imdb.images.label(imdb.selected, resdb.video_ids));
% info.AP_tag = compute_average_precision(prob, gts);

% compute the prec@k
fprintf('Computing prec@K...\n');
% info.prec_at_8 = compute_precision_at_k(prob, gts, 'k', 8);
% info.prec_at_16 = compute_precision_at_k(prob, gts, 'k', 16);
info.prec_at_32 = compute_precision_at_k(prob, gts, 'k', 32);

% computing the adjusted prec@k
vetted_labels = load_vetted_labels();
fprintf('Computing adjusted prec@K...\n');
vetted_labels = vetted_labels(imdb.selected, resdb.video_ids);
vetted_gts = gts;
vetted_gts(vetted_labels==2) = 1;
vetted_gts(vetted_labels==-2) = 0;
% info.adjusted_prec_at_8 = compute_precision_at_k(prob,...
%   vetted_gts, 'k', 8);
% info.adjusted_prec_at_16 = compute_precision_at_k(prob,...
%   vetted_gts, 'k', 16);
info.adjusted_prec_at_32 = compute_precision_at_k(prob,...
  vetted_gts, 'k', 32);

info_path = fullfile(opts.expDir, 'info.mat');
fprintf('saving the AP info to %s\n', info_path);
save(info_path, '-struct', 'info');

function vetted_labels = load_vetted_labels()
vetted_labels_train = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load('/home/phuc/Research/yaromil/vetted_labels_test.mat');
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

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