function run_mm_at_dual(imdb, model_A, resdb_A, resdb_B, vetted_labels, varargin)
% run multi-model active testing - dual version
% In this version, after each vetting period, we are also retraining A.
% Args:
%   imdb: the common imdb
%   model_A: struct, have the vetted indeces + estimator_A
%   resdb_A: the resdb for A
%   resdb_A: the resdb for B
%   vetted_labels: the whole vetted_labels
opts.k_budget = 48;
opts.k_evaluation = 48;
opts.k_budget_batch = 4;
opts.search_mode = 'global';
opts = vl_argparse(opts, varargin);

% load the true precision
lstruct = load('active_testing/true_precision_at_48.mat');
true_precisions = lstruct.prec;

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
opts.tag_indeces = find(tag_indeces_bin);
opts.tag_indeces_1000 = find(tag_indeces_b1000_bin);
opts.tag_set_1000 = tag_set_1000;
opts.tag_set_4000 = imdb.classes.name;
opts.imdb = imdb;

% load the vetted labels
vetted_labels = vetted_labels(:, resdb_A.video_ids);
observed_label = imdb.images.label(:, resdb_A.video_ids);
label_set = struct();
label_set.vetted_labels = vetted_labels(imdb.tags_to_train, :);
label_set.observed_label = observed_label(imdb.tags_to_train, :);

% load the classifier scores
prob_A = resdb_A.fc1000.outputs(imdb.tags_to_train, :);
prob_B = resdb_B.fc1000.outputs(imdb.tags_to_train, :);

num_tags = numel(tags);
total_budget = opts.k_budget*num_tags;
batch_budget = opts.k_budget_batch*num_tags;
save_file_name = sprintf('active_testing/mm_dual_res_adaptive_%s.mat', opts.search_mode);
% train the initial model using the vetted_labels from A
vetted_examples = model_A.to_use_vetted(imdb.tags_to_train, :);
res = struct(); res.infoA(1).to_use_vetted = sparse(vetted_examples);
res.info(1).to_use_vetted = sparse(vetted_examples); % the visible part of the vetted matrix
res.info(1).current_budget = 0;
iter = 1;
save(save_file_name, '-struct', 'res');

fprintf('Training the first estimator B\n');
estimator_A = model_A.estimator;
estimator_B = train_cnn(imdb, label_set, vetted_examples, resdb_B, 'modelB', opts);
precision_B = get_precision(estimator_B, res.info(1).to_use_vetted,...
  label_set, prob_B, opts);
precision_A = get_precision(estimator_A, res.info(1).to_use_vetted,...
  label_set, prob_A, opts);
res.info(1).precision_B = precision_B;
res.info(1).precision_A = precision_A;

% for plotting
done = false;
while ~done
  %res.info(iter).current_budget <= total_budget - batch_budget
  % query the least confident according to the estimator B
  fprintf('iter %d: querying the least confident examples\n', iter);
  new_vetted_set = query_least_confident(estimator_A, estimator_B,...
    res.info(iter).to_use_vetted, label_set, prob_A, prob_B, opts);
  
  res.info(iter+1).to_use_vetted = new_vetted_set; % vetting
  res.info(iter+1).current_budget = res.info(iter+1).current_budget + batch_budget;
  
  % retrain the model
  fprintf('iter %d: retraining the estimator\n', iter);
  estimator_B = train_cnn(imdb, label_set,...
    res.info(iter+1).to_use_vetted, resdb_B, 'modelB', opts);
  estimator_A = train_cnn(imdb, label_set,...
    res.info(iter+1).to_use_vetted, resdb_A, 'modelA', opts);
  
  precision_B = get_precision(estimator_B, res.info(iter+1).to_use_vetted,...
    label_set, prob_B, opts);
  precision_A = get_precision(estimator_A, res.info(1).to_use_vetted,...
    label_set, prob_A, opts);
  res.info(iter+1).precision_B = precision_B;
  res.info(iter+1).precision_A = precision_A;
  
  iter = iter +1;
  save(save_file_name, '-struct', 'res');
  if res.info(iter).current_budget >  (total_budget - batch_budget)
    fprintf('Not enough budget left, exiting...\n')
    done = true;
  end
end

% -------------------------------------------------------------------------
function to_use_vetted = query_least_confident(net_A, net_B,...
  to_use_vetted, label_set, prob_A, prob_B, opts)
% Args:
%   net_budget: the current estimator
%   to_use_vetted: the current vetted.
%   vetted_labels: the whole vetted labels, everything
%   observed_labels: the observed labels
%   prob: the classifier scores, 1000xN
%   tag_indeces: the tag_indeces that we care about
%   imdb: the current imdb
%   opts: the options
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
% observed_label = label_set.observed_label; % 1Kx160K

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);
all_videos_A_per_tag = cell(num_tag, 1);
all_videos_B_per_tag = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds_A = prob_A(tag_index_1000, :);
  [~, order_A] = sort(preds_A, 'descend');
  topK_videoid = order_A(1:opts.k_evaluation);
  all_videos_A_per_tag{index} = topK_videoid;
  
  preds_B = prob_B(tag_index_1000, :);
  [~, order_B] = sort(preds_B, 'descend');
  topK_videoid = order_B(1:opts.k_evaluation);
  all_videos_B_per_tag{index} = topK_videoid;
end
all_videos_A = cat(2, all_videos_A_per_tag{:});
all_videos_A = unique(all_videos_A);
all_videos_B = cat(2, all_videos_B_per_tag{:});
all_videos_B = unique(all_videos_B);
%all_videos = [all_videos_A all_videos_B];
%all_videos = unique(all_videos);

% getting the pos prob for A
estimator_scores_A = get_posterior_pos(label_set, prob_A, all_videos_A, net_A);

% getting the pos prob for B
estimator_scores_B = get_posterior_pos(label_set, prob_B, all_videos_B, net_B);

% find the examples
lookup_indeces = cell(num_tag, 1);
certainty_all = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  topK_videoid_A = all_videos_A_per_tag{index};
  topK_videoid_B = all_videos_B_per_tag{index};
  
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  unvetted_videoid_A = setdiff(topK_videoid_A, vetted_videoid);
  unvetted_videoid_B = setdiff(topK_videoid_B, vetted_videoid);
  [~,~,ib_A] = intersect(unvetted_videoid_A, all_videos_A);
  [~,~,ib_B] = intersect(unvetted_videoid_B, all_videos_B);
  pos_prob_A = estimator_scores_A(tag_index_1000, ib_A);
  pos_prob_B = estimator_scores_B(tag_index_1000, ib_B);
  unvetted_videoid = [unvetted_videoid_A unvetted_videoid_B];
  
  certainty_all{index} = [pos_prob_A pos_prob_B];
  lookup_indeces{index} = [tag_index_1000 * ones(numel(unvetted_videoid), 1) unvetted_videoid'];
end
lookup_indeces = cat(1, lookup_indeces{:});
certainty_all = cat(2, certainty_all{:});
assert(size(lookup_indeces, 1) == size(certainty_all,2));

[~, order] = sort(abs(certainty_all - 0.5));
to_take = min(opts.k_budget_batch*75, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
to_use_vetted(sub) = 1;

% -------------------------------------------------------------------------
function estimator_scores = get_posterior_pos(label_set,...
  prob, video_ids, net)
% -------------------------------------------------------------------------
observed_label = label_set.observed_label;
observed_lbls = full(observed_label(:, video_ids));
scores = vl_nnsigmoid(prob(:, video_ids));
inputs = [scores; observed_lbls]; inputs = permute(inputs, [3 4 1 2]);
inputs = gpuArray(single(inputs));
labels = single(rand(size(scores))); labels = permute(labels, [3 4 1 2]);
labels = gpuArray(single(labels));
net.move('gpu'); net.vars(2).precious = true;
net.eval({'input', inputs, 'labels', labels});
fc1 = gather(net.vars(2).value); fc1 = permute(fc1, [3 4 1 2]);
estimator_scores = vl_nnsigmoid(fc1); % 1Kx160K


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
tag_indeces = opts.tag_indeces;
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
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
scores = vl_nnsigmoid(prob(:, all_videos));
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
function net=train_cnn(imdb, label_set, to_use_vetted, resdb, model_name, opts)
% -------------------------------------------------------------------------
vetted_labels = label_set.vetted_labels;
observed_label = label_set.observed_label;
current_budget = full(sum(sum(to_use_vetted)));
visible_vetted_labels = double(vetted_labels) .* to_use_vetted;
exp_name = sprintf('mm_dual_adaptive_%d_%s_%s', current_budget, opts.search_mode, model_name);
path_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
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
  fc1000 = resdb.fc1000.outputs(new_imdb.tags_to_train, :);
  fc1000_path = fullfile(path_dir, 'fc1000.mat');
  save(fc1000_path, 'fc1000');
  fprintf('Done\n');

  % run the learner
  [net,~]=run_train_language(exp_name, 1);
else
  % load the model
  [epoch, iter] = findLastCheckpoint(path_dir);
  model_path = fullfile(path_dir,...
    sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));
  net = load(model_path);
  net = dagnn.DagNN.loadobj(net.net) ;
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