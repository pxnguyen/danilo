function run_sm_at_mcn(imdb, resdb, vetted_labels, varargin)
% run multi-model active testing basic version
% Args:
%   imdb: the common imdb
%   model_A: struct, have the vetted indeces + estimator_A
%   resdb_A: the resdb for A
%   resdb_A: the resdb for B
%   vetted_labels: the whole vetted_labels
opts.max_budget = 48*75;
opts.batch_budget = 4*75;
opts.k_evaluation = 48;
opts.estimator = 'learner';
opts.search_mode = 'global';
opts.cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
opts.gpu = 1;
opts = vl_argparse(opts, varargin);

% load the true precision
% lstruct = load('active_testing/true_precision_at_48.mat');
% true_precisions = lstruct.prec;
otps.strategy = 'mcn';

fid = fopen('active_testing/tags.list');
tag_set_1000 = imdb.classes.name(imdb.selected);
tags = textscan(fid, '%s\n'); tags = tags{1};
tag_indeces_bin = false(4000, 1);
tag_indeces_b1000_bin = false(1000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(imdb.classes.name, tags{index});
  tag_indeces_bin(tag_index) = true;
  
  tag_index = strcmp(tag_set_1000, tags{index});
  tag_indeces_b1000_bin(tag_index) = true;
end
opts.tag_indeces_1000 = find(tag_indeces_b1000_bin);
opts.tag_set_1000 = tag_set_1000;
opts.imdb = imdb;

% vetted labels
vetted_labels = vetted_labels(imdb.tags_to_train, resdb.video_ids);
observed_label = imdb.images.label(imdb.tags_to_train, resdb.video_ids);
label_set.vetted_labels = vetted_labels;
label_set.observed_label = observed_label;

% load the classifier scores
prob = resdb.fc1000.outputs(imdb.tags_to_train, :);

total_budget = opts.max_budget;
batch_budget = opts.batch_budget;
save_file_name = sprintf('active_testing/res_sm_mcn_%s_%s.mat', opts.estimator, opts.search_mode);
rng(1);
res = struct();
vetted_examples = mcn(prob, zeros(size(vetted_labels)), label_set, opts);
res.info(1).to_use_vetted = sparse(vetted_examples); % the visible part of the vetted matrix
res.info(1).current_budget = sum(sum(vetted_examples));

if strcmp(opts.estimator, 'learner')
  estimator = train_cnn(imdb, label_set, vetted_examples, resdb, opts);
  precision = get_precision(estimator, res.info(1).to_use_vetted,...
   label_set, prob, opts);
else
  precision = get_precision_naive(res.info(1).to_use_vetted,...
   label_set, prob, opts);
end
res.info(1).precision = precision;
save(save_file_name, '-struct', 'res');

% for plotting
iter = 2;
done = false;
while ~done
  fprintf('iter %d: querying random examples\n', iter);
  new_vetted_set = mcn(prob, res.info(iter-1).to_use_vetted, label_set, opts);
  res.info(iter).to_use_vetted = new_vetted_set; % vetting
  res.info(iter).current_budget = res.info(iter-1).current_budget + batch_budget;

  % use this to decouple the effect of the sample vs estimator
  if strcmp(opts.estimator, 'learner')
    precision_before_retrained = get_precision(estimator, new_vetted_set,...
      label_set, prob, opts);
    res.info(iter).precision_before_retrained = precision_before_retrained;
  else
    precision_before_retrained = get_precision_naive(res.info(iter).to_use_vetted,...
     label_set, prob, opts);
   res.info(iter).precision_before_retrained = precision_before_retrained;
  end

  % retrain the model
  fprintf('iter %d: retraining estimator\n', iter);
  switch opts.estimator
    case 'learner'
    estimator = train_cnn(imdb, label_set,...
      res.info(iter).to_use_vetted, resdb, opts);
    precision = get_precision(estimator, res.info(iter).to_use_vetted,...
      label_set, prob, opts);
  else
    precision = get_precision_naive(res.info(iter).to_use_vetted,...
     label_set, prob, opts);
  end
  res.info(iter).precision = precision;
  
  fprintf('iter %d: cur_budget: %d total %d batch %d\n', iter,...
    res.info(iter).current_budget,...
    total_budget, batch_budget);

  save(save_file_name, '-struct', 'res');
  if res.info(iter).current_budget >  total_budget - batch_budget
    fprintf('Not enough budget left, exiting...\n')
    done = true;
  end
  iter = iter +1;
end

% -------------------------------------------------------------------------
function precisions = get_precision_naive(to_use_vetted, label_set,...
  prob, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index_1000, vetted_videoid)>1);

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prob = observed_label(tag_index_1000, unvetted_videoid);
  unvetted_count = full(sum(pos_prob));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function new_use_vetted=mcn(prob, to_use_vetted, label_set, opts)
% -------------------------------------------------------------------------
observed_label = label_set.observed_label;
vetted_labels = label_set.vetted_labels;
tag_indeces_1000 = opts.tag_indeces_1000;
num_tag = numel(tag_indeces_1000);
new_use_vetted = to_use_vetted; % all the vetted pairs
lookup_indeces = cell(num_tag, 1);
scores_all = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);

  cat_prob = prob(tag_index_1000, :);
  [shortlist_scores, shortlist_order] = sort(cat_prob, 'descend');
  shortlist_scores = shortlist_scores(1:opts.k_evaluation);
  topk_videoids = shortlist_order(1:opts.k_evaluation);

  negative = ~observed_label(tag_index_1000, :);
  not_already_vetted = ~to_use_vetted(tag_index_1000, :);
  has_vetted_label = abs(vetted_labels(tag_index_1000, :)) > 1;

  condition = (negative & has_vetted_label & not_already_vetted);
  [~, ~, ib] = intersect(find(condition), topk_videoids);
  videoids_fit_scores = shortlist_scores(ib);
  scores_all{index} = videoids_fit_scores;

  indeces = [tag_index_1000 * ones(numel(videoids_fit_scores), 1) shortlist_order(ib)'];
  lookup_indeces{index} = indeces;
end
scores_all = cat(2, scores_all{:});
lookup_indeces = cat(1, lookup_indeces{:});

[~, order] = sort(scores_all, 'descend');
to_take = min(opts.batch_budget, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
new_use_vetted(sub) = 1;

% -------------------------------------------------------------------------
function precisions = get_precision(net, to_use_vetted, label_set,...
  prob, opts)
% Args:
%   net_budget: the current estimator
%   to_use_vetted: the current vetted.
%   label_set: contain the observed+vetted labels
%   prob: the classifier scores, 1000xN
%   tag_indeces: the tag_indeces that we care about
%   imdb: the current imdb
%   opts: the options
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);
all_videos = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');
  topK_videoid = order(1:opts.k_evaluation);
  all_videos{index} = topK_videoid;
end
all_videos = cat(2, all_videos{:}); all_videos = unique(all_videos);

% making the features
observed_lbls = full(observed_label(:, all_videos));
fc1000_mean = net.meta.normalization.fc1000_mean;
fc1000_std = net.meta.normalization.fc1000_std;
scores = (prob(:, all_videos) - fc1000_mean)./fc1000_std;
inputs = [scores; observed_lbls]; inputs = permute(inputs, [3 4 1 2]);
inputs = gpuArray(single(inputs));
labels = single(rand(size(scores))); labels = permute(labels, [3 4 1 2]);
labels = gpuArray(single(labels));
net.move('gpu'); net.vars(2).precious = true;
net.eval({'input', inputs, 'labels', labels});
fc1 = gather(net.vars(2).value); fc1 = permute(fc1, [3 4 1 2]);
estimator_scores = vl_nnsigmoid(fc1); % 1Kx160K

% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index_1000, vetted_videoid)>1);

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  [~,~,ib] = intersect(unvetted_videoid, all_videos);
  pos_prob = estimator_scores(tag_index_1000, ib);
  unvetted_count = sum(pos_prob);
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function net=train_cnn(imdb, label_set, to_use_vetted, resdb, opts)
% -------------------------------------------------------------------------
vetted_labels = label_set.vetted_labels;
observed_label = label_set.observed_label;
current_budget = full(sum(sum(to_use_vetted)));
visible_vetted_labels = double(vetted_labels) .* to_use_vetted;
exp_name = sprintf('single_model_mcn_%d', current_budget);
path_dir = fullfile(opts.cnn_exp, exp_name);
if ~exist(path_dir, 'dir'); mkdir(path_dir); end
if isempty(dir(fullfile(path_dir, 'net-*.mat')))
  new_imdb = imdb;
  new_imdb.images.label = sparse(4000, numel(resdb.video_ids));
  new_imdb.images.label(imdb.tags_to_train, :) = observed_label;
  new_imdb.images.name = new_imdb.images.name(resdb.video_ids);
  new_imdb.images.set = ones(1, numel(resdb.video_ids));
  new_imdb.images = rmfield(new_imdb.images, 'vetted_label');

  % combined the features
  vis_label = sparse(4000, numel(resdb.video_ids));
  vis_label(imdb.tags_to_train, :) = visible_vetted_labels;
  
  new_imdb.images.combined_label = new_imdb.images.label;
  new_imdb.images.combined_label(vis_label==2) = 1.0; % slow
  new_imdb.images.combined_label(vis_label==-2) = 0.0; % slow

  % save the imdb
  imdb_name = sprintf('%s_imdb.mat', exp_name);
  imdb_path = fullfile(path_dir, imdb_name);
  save(imdb_path, '-struct', 'new_imdb');

  % save the scores - THIS IS SLOW, can just be copy from somewhere.
  fprintf('Saving predictions...')
  if ~exist('active_testing/fc1000.mat')
    fc1000_norm = struct();
    fc1000 = resdb.fc1000.outputs(new_imdb.tags_to_train, :);
    fc1000_mean = mean(fc1000, 2);
    fc1000_std = std(fc1000, [], 2);
    fc1000 = (fc1000 - fc1000_mean)./fc1000_std;
    fc1000_path = fullfile(path_dir, 'fc1000.mat');
    save(fc1000_path, 'fc1000');

    fc1000_norm_path = fullfile(path_dir, 'fc1000_norm.mat');
    fc1000_norm.mean = fc1000_mean;
    fc1000_norm.std = fc1000_std;
    save(fc1000_norm_path, '-struct', 'fc1000_norm');
    fprintf('Done\n');
  else
    copyfile('active_testing/fc1000.mat', fullfile(path_dir, 'fc1000.mat'));
    copyfile('active_testing/fc1000_norm.mat', fullfile(path_dir, 'fc1000_norm.mat'));
    
    fc1000_norm = load(fullfile(path_dir, 'fc1000_norm.mat'));
  end

  % run the learner
  [net,~]=run_train_language(exp_name, opts.gpu);
  net.meta.normalization.fc1000_mean = fc1000_norm.mean;
  net.meta.normalization.fc1000_std = fc1000_norm.std;
else
  % load the model
  [epoch, iter] = findLastCheckpoint(path_dir);
  model_path = fullfile(path_dir,...
    sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));
  net = load(model_path);
  net = dagnn.DagNN.loadobj(net.net) ;
  fc1000_mean_path = fullfile(path_dir, 'fc1000_norm.mat');
  norm = load(fc1000_mean_path);
  net.meta.normalization.fc1000_mean = norm.mean;
  net.meta.normalization.fc1000_std = norm.std;
end

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