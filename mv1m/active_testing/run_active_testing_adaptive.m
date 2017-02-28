function [precisions, delta]=run_active_testing_adaptive(resdb, vetted_labels, varargin)
opts.k_budget = 48;
opts.k_evaluation = 48;
opts.k_budget_batch = 4;
opts.search_mode = 'local';
opts = vl_argparse(opts, varargin);

% load aria imdb
imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');

% load the true precision
lstruct = load('active_testing/true_precision_at_48.mat');
true_precisions = lstruct.prec;

fid = fopen('active_testing/tags.list');
tags = textscan(fid, '%s\n'); tags = tags{1};
tag_indeces_bin = false(4000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(imdb.classes.name, tags{index});
  tag_indeces_bin(tag_index) = true;
end
tag_indeces = find(tag_indeces_bin);

% load the vetted labels
vetted_labels = vetted_labels(:, resdb.video_ids);
observed_label = imdb.images.label(:, resdb.video_ids);
prob = resdb.fc1000.outputs;

num_tags = numel(tags);
total_budget = opts.k_budget*num_tags;
batch_budget = opts.k_budget_batch*num_tags;
save_file_name = sprintf('active_testing/res_adaptive_%s.mat', opts.search_mode);
% query random
if ~exist(save_file_name, 'file')
  rng(1); % reproducable
  to_use_vetted=random_sampling(tag_indeces, prob, vetted_labels, opts);
  net_estimator = train_cnn(imdb, observed_label, vetted_labels,...
    to_use_vetted, resdb, opts); % train the init model (CNN)
  precisions_all = cell(1, 1);
  res = struct(); res.data = struct();
  res.info(1).to_use_vetted = sparse(to_use_vetted);
  res.info(1).current_budget = sum(sum(to_use_vetted==1));
  save(save_file_name, '-struct', 'res');
  iter = 1;
else
  res = load(save_file_name);
  net_estimator = train_cnn(imdb, observed_label, vetted_labels,...
    res.info(end).to_use_vetted, resdb, opts);
  iter = 1;
end

% for plotting
while res.info(iter).current_budget <= total_budget - batch_budget
  % query the least confident according to the estimator.
  fprintf('iter %d: querying the least confident examples\n', iter);
  [new_vetted_set, precisions] = query_least_confident(net_estimator,...
    res.info(iter).to_use_vetted, vetted_labels, observed_label, prob, tag_indeces, imdb, opts);
  res.info(iter).precisions = precisions;
  res.info(iter).delta = mean(abs(precisions(:) - true_precisions(:)));
  
  res.info(iter+1).to_use_vetted = new_vetted_set;
  res.info(iter+1).current_budget = res.info(iter).current_budget + batch_budget;
  assert(sum(sum(res.info(iter+1).to_use_vetted)) == res.info(iter+1).current_budget);
  % retrain the model
  fprintf('iter %d: retraining the estimator\n', iter);
  net_estimator = train_cnn(imdb, observed_label, vetted_labels,...
    res.info(iter+1).to_use_vetted, resdb, opts);
  precisions_all{iter} = precisions;
  iter = iter +1;
  save('active_testing/res_adaptive.mat', '-struct', 'res');
end

function [to_use_vetted, precisions] = query_least_confident(net_budget,...
  to_use_vetted, vetted_labels, observed_label, prob,...
  tag_indeces, imdb, opts)
all_videos = cell(numel(tag_indeces), 1);
for index = 1:numel(tag_indeces)
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');
  topK_videoid = order(1:opts.k_evaluation);
  all_videos{index} = topK_videoid;
end
all_videos = cat(2, all_videos{:}); all_videos = unique(all_videos);

observed_lbls = full(observed_label(imdb.tags_to_train, all_videos));
scores = vl_nnsigmoid(prob(imdb.tags_to_train, all_videos));

inputs = [scores; observed_lbls]; inputs = permute(inputs, [3 4 1 2]);
inputs = gpuArray(single(inputs));
labels = single(rand(size(scores))); labels = permute(labels, [3 4 1 2]);
labels = gpuArray(single(labels));

net_budget.move('gpu'); net_budget.vars(2).precious = true;
net_budget.eval({'input', inputs, 'labels', labels});

fc1 = gather(net_budget.vars(2).value); fc1 = permute(fc1, [3 4 1 2]);
fc1 = vl_nnsigmoid(fc1);

full_fc1 = zeros(size(prob));
full_fc1(imdb.tags_to_train, all_videos) = fc1;

% find the examples and computing the precision
lookup_indeces = cell(numel(tag_indeces), 1);
certainty_all = cell(numel(tag_indeces), 1);
precisions = zeros(numel(tag_indeces), 1);
for index = 1:numel(tag_indeces)
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid)>1);

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prob = full_fc1(tag_index, unvetted_videoid);
  unvetted_count = sum(pos_prob);
  
  certainty_all{index} = pos_prob;
  lookup_indeces{index} = [tag_index * ones(numel(unvetted_videoid), 1) unvetted_videoid'];
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end
% TODO: need to handle the case for local
lookup_indeces = cat(1, lookup_indeces{:});
certainty_all = cat(2, certainty_all{:});

[~, order] = sort(abs(certainty_all - 0.5));
to_take = min(opts.k_budget_batch*75, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
to_use_vetted(sub) = 1;

% -------------------------------------------------------------------------
function net=train_cnn(imdb, observed_label, vetted_labels, to_use_vetted, resdb, opts)
% -------------------------------------------------------------------------
current_budget = full(sum(sum(to_use_vetted)));
visible_vetted_labels = double(vetted_labels) .* to_use_vetted;
exp_name = sprintf('adaptive_%d_%s', current_budget, opts.search_mode);
path_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
if ~exist(path_dir, 'dir'); mkdir(path_dir); end
if isempty(dir(fullfile(path_dir, 'net-*.mat')))
  new_imdb = imdb;
  new_imdb.images.label = observed_label;
  new_imdb.images.name = new_imdb.images.name(resdb.video_ids);
  new_imdb.images.set = ones(1, numel(resdb.video_ids));
  new_imdb.images = rmfield(new_imdb.images, 'vetted_label');

  % combined the features
  new_imdb.images.combined_label = new_imdb.images.label;
  new_imdb.images.combined_label(visible_vetted_labels==2) = 1.0; % slow
  new_imdb.images.combined_label(visible_vetted_labels==-2) = 0.0; % slow

  % save the imdb
  imdb_name = sprintf('%s_imdb.mat', exp_name);
  imdb_path = fullfile(path_dir, imdb_name);
  save(imdb_path, '-struct', 'new_imdb');

  % save the scores
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
function to_use_vetted=random_sampling(tag_indeces, prob, vetted_labels, opts)
% -------------------------------------------------------------------------
to_use_vetted = zeros(size(vetted_labels)); % all the vetted pairs
for index = 1:numel(tag_indeces)
  tag_index = tag_indeces(index);
  has_vetted_label = abs(vetted_labels(tag_index, :)) > 1;

  cat_prob = prob(tag_index, :);
  [~, order] = sort(cat_prob, 'descend');
  order = order(1:opts.k_evaluation);
  videoids_fit_condition = find(has_vetted_label);
  
  vetted_shortlist = intersect(order, videoids_fit_condition);
  random_sample = randperm(numel(vetted_shortlist));
  random_sample = random_sample(min(1:opts.k_budget_batch, numel(random_sample)));
  vetted_shortlist = vetted_shortlist(random_sample);

  to_use_vetted(tag_index, vetted_shortlist) = 1;
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