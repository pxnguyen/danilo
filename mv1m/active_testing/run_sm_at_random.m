function run_sm_at_random(imdb, resdb, vetted_labels, varargin)
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
opts.search_mode = 'global';
opts.strategy = 'random';
opts.estimator = 'learner';
opts.cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
opts.gpu = 1;
opts = vl_argparse(opts, varargin);

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
save_file_name = sprintf('active_testing/res_sm_%s_%s_%s.mat', opts.strategy,...
  opts.estimator, opts.search_mode);
rng(1);
res = struct();
res.name = sprintf('%s-%s', opts.strategy, opts.estimator);
vetted_examples = random_sampling(prob, zeros(size(vetted_labels)), vetted_labels, opts);
res.info(1).to_use_vetted = sparse(vetted_examples); % the visible part of the vetted matrix
res.info(1).current_budget = sum(sum(vetted_examples));

opts = update_params(label_set, res.info(1).to_use_vetted, resdb, prob, opts);
precision = get_precision_all(res.info(1).to_use_vetted, label_set, prob, opts);

res.info(1).precision = precision;
res.info(1).opts = opts;
save(save_file_name, '-struct', 'res');

% for plotting
iter = 2;
done = false;
while ~done
  fprintf('iter %d: querying random examples\n', iter);
  switch opts.strategy
    case 'random'
      new_vetted_set = random_sampling(prob, res.info(iter-1).to_use_vetted, vetted_labels, opts);
    case 'mcn'
      new_vetted_set = mcn(prob, res.info(iter-1).to_use_vetted, label_set, opts);
    case 'adaptive'
      new_vetted_set = most_confused(prob, res.info(iter-1).to_use_vetted, label_set, opts);
    otherwise
  end
  res.info(iter).to_use_vetted = new_vetted_set; % vetting
  res.info(iter).current_budget = full(sum(sum(res.info(iter).to_use_vetted)));
  % use this to decouple the effect of the sample vs estimator
  precision_before_retrained = get_precision_all(res.info(iter).to_use_vetted, label_set, prob, opts);
  res.info(iter).precision_before_retrained = precision_before_retrained;
  % retrain the model
  fprintf('iter %d: retraining estimator\n', iter);
  opts = update_params(label_set, res.info(iter).to_use_vetted, resdb, prob, opts);
  res.info(iter).opts = opts;
  precision = get_precision_all(res.info(iter).to_use_vetted, label_set, prob, opts);
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
function opts=update_params(label_set, to_use_vetted, resdb, prob, opts)
% -------------------------------------------------------------------------
switch opts.estimator
  case 'learner'
    opts.estimator = train_cnn(imdb, label_set,...
      to_use_vetted, resdb, opts);
  case 'prior'
    opts.priors = update_prior(to_use_vetted, label_set, prob, opts);
  case 'prior2'
    opts.priors = update_prior2(to_use_vetted, label_set, prob, opts);
  case 'prior3'
    opts.priors = update_prior3(to_use_vetted, label_set, prob, opts);
  case 'naive'
  otherwise
    error('Unrecognized estimator');
end

% -------------------------------------------------------------------------
function precisions = get_precision_all(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
switch opts.estimator
  case 'learner'
    estimator = opts.estimator;
    precisions = get_precision(estimator, new_vetted_set,...
      label_set, prob, opts);
  case 'prior'
    priors = opts.priors;
    precisions = get_precision_prior_based(to_use_vetted,...
     label_set, prob, priors, opts);
  case 'prior2'
    priors = opts.priors;
    precisions = get_precision_prior2_based(to_use_vetted,...
     label_set, prob, priors, opts);
  case 'prior3'
    priors = opts.priors;
    precisions = get_precision_prior3_based(to_use_vetted,...
      label_set, prob, priors, opts);
  case 'lasso'
    priors = opts.priors;
    precisions = get_precision_prior3_based(to_use_vetted,...
      label_set, prob, priors, opts);
  case 'naive'
    precisions = get_precision_naive(to_use_vetted, label_set, prob, opts);
  otherwise
    error('Unrecognized estimator');
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
function new_use_vetted=random_sampling(prob, to_use_vetted, vetted_labels, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
num_tag = numel(tag_indeces_1000);
new_use_vetted = to_use_vetted; % all the vetted pairs
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  has_vetted_label = abs(vetted_labels(tag_index_1000, :)) > 1;
  already_vetted = to_use_vetted(tag_index_1000, :);
  condition = (has_vetted_label & ~already_vetted);

  cat_prob = prob(tag_index_1000, :);
  [~, order] = sort(cat_prob, 'descend');
  topk_videoid = order(1:opts.k_evaluation);
  videoids_fit_condition = find(condition);
  
  vetted_shortlist = intersect(topk_videoid, videoids_fit_condition);
  random_sample = randperm(numel(vetted_shortlist));
  random_sample = random_sample(min(1:opts.batch_budget/num_tag,...
    numel(random_sample)));
  vetted_shortlist = vetted_shortlist(random_sample);

  new_use_vetted(tag_index_1000, vetted_shortlist) = 1;
end

% -------------------------------------------------------------------------
function to_use_vetted=most_confused(prob, to_use_vetted, label_set, opts)
% -------------------------------------------------------------------------
priors = opts.priors;
tag_indeces_1000 = opts.tag_indeces_1000;
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);

% find the examples and computing the precision
lookup_indeces = cell(num_tag, 1);
certainty_all = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prob = zeros(1, numel(unvetted_videoid));
  pos = boolean(full(observed_label(tag_index_1000, unvetted_videoid)));
  neg = boolean(full(~observed_label(tag_index_1000, unvetted_videoid)));
  pos_prob(pos) = priors.flip_priors(index, 2);
  pos_prob(neg) = priors.flip_priors(index, 1);
  
  certainty_all{index} = pos_prob;
  lookup_indeces{index} = [tag_index_1000 * ones(numel(unvetted_videoid), 1) unvetted_videoid'];
end

lookup_indeces = cat(1, lookup_indeces{:});
certainty_all = cat(2, certainty_all{:});

[~, order] = sort(abs(certainty_all - 0.5));
to_take = min(opts.batch_budget, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
to_use_vetted(sub) = 1;

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
function priors = update_prior(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = label_set.vetted_labels; % 1Kx160K
visible_vetted_labels = single(vetted_labels) .* full(to_use_vetted);
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);
% find the examples and computing the precision
priors = struct;
priors.flip_priors = zeros(num_tag, 2); % p(y|z)
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  topk_vetted_videoid = intersect(vetted_videoid, topk_videoid);
  y = observed_label(tag_index_1000, topk_vetted_videoid);
  z = visible_vetted_labels(tag_index_1000, topk_vetted_videoid) > 1;
  priors.flip_priors(index, 1) = (sum(y==0 & z==1)+0.1)/(sum(y==0)+0.1);
  priors.flip_priors(index, 2) = (sum(y== 1 & z==1)+0.1)/(sum(y==1)+0.1);
end

% -------------------------------------------------------------------------
function priors = update_prior3(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = label_set.vetted_labels; % 1Kx160K
visible_vetted_labels = single(vetted_labels) .* full(to_use_vetted);
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces_1000);
% find the examples and computing the precision
priors = struct;
priors.beta = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));
  topk_vetted_videoid = intersect(vetted_videoid, topk_videoid);
  
  y = full(observed_label(tag_index_1000, topk_vetted_videoid));
  s = prob(tag_index_1000, topk_vetted_videoid);
  %inputs = [y' ones(numel(y), 1)];
  inputs = [s' y' ones(numel(y), 1)];
  z = full(visible_vetted_labels(tag_index_1000, topk_vetted_videoid) > 1);
  priors.beta{index} = regress(z',inputs);
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior_based(to_use_vetted, label_set,...
  prob, priors, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
vetted_labels = single(label_set.vetted_labels) .* full(to_use_vetted); % 1Kx160K
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
  pos_prior = observed_label(tag_index_1000, unvetted_videoid) * priors.flip_priors(index, 2);
  neg_prior = ~observed_label(tag_index_1000, unvetted_videoid) * priors.flip_priors(index, 1);
  unvetted_count = full(sum(pos_prior) + sum(neg_prior));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior2_based(to_use_vetted, label_set,...
  prob, priors, opts)
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
  q = priors.q(index);
  p = priors.p(index);
  r = priors.r(index);
  eps = 1e-3;
  pos = observed_label(tag_index_1000, unvetted_videoid);
  pos_prior = pos * (q*r+eps)/(q*r+p*(1-r)+eps);
  
  neg = ~observed_label(tag_index_1000, unvetted_videoid);
  neg_prior = neg * ((1-q)*r+eps)/((1-p)*(1-r) + (1-q)*r+eps);
  unvetted_count = full(sum(pos_prior) + sum(neg_prior));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior3_based(to_use_vetted, label_set,...
  prob, priors, opts)
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
  beta = priors.beta{index};
  y = full(observed_label(tag_index_1000, unvetted_videoid));
  s = prob(tag_index_1000, unvetted_videoid);
  %inputs = [y' ones(numel(y), 1)];
  inputs = [s' y' ones(numel(y), 1)];
  pos_prob = inputs*beta;
  unvetted_count = sum(pos_prob);
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function new_vetted_set = most_confused_prior(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
tag_indeces_1000 = opts.tag_indeces_1000;
num_tag = numel(tag_indeces_1000);
observed_label = label_set.observed_label; % 1Kx160K

new_vetted_set = to_use_vetted;
all_videos = cell(numel(tag_indeces_1000), 1);
for index = 1:numel(tag_indeces_1000)
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');
  topK_videoid = order(1:opts.k_evaluation);
  all_videos{index} = topK_videoid;
end
all_videos = cat(2, all_videos{:}); all_videos = unique(all_videos);

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
estimator_scores = vl_nnsigmoid(fc1);

% find the examples and computing the precision
lookup_indeces = cell(num_tag, 1);
certainty_all = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);

  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index_1000, :));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  [~,~,ib] = intersect(unvetted_videoid, all_videos);
  pos_prob = estimator_scores(tag_index_1000, ib);

  certainty_all{index} = pos_prob;
  lookup_indeces{index} = [tag_index_1000 * ones(numel(unvetted_videoid), 1) unvetted_videoid'];
end
lookup_indeces = cat(1, lookup_indeces{:});
certainty_all = cat(2, certainty_all{:});

[~, order] = sort(abs(certainty_all - 0.5));
to_take = min(opts.batch_budget, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
new_vetted_set(sub) = 1;