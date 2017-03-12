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
opts.cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
opts.gpu = 1;
opts = vl_argparse(opts, varargin);

% load the true precision
% lstruct = load('active_testing/true_precision_at_48.mat');
% true_precisions = lstruct.prec;

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
save_file_name = sprintf('active_testing/res_sm_random_%s.mat', opts.search_mode);
rng(1);
res = struct();
vetted_examples = random_sampling(prob, zeros(size(vetted_labels)), vetted_labels, opts);
res.info(1).to_use_vetted = sparse(vetted_examples); % the visible part of the vetted matrix
res.info(1).current_budget = sum(sum(vetted_examples));

estimator = train_cnn(imdb, label_set, vetted_examples, resdb, opts);
precision = get_precision(estimator, res.info(1).to_use_vetted,...
 label_set, prob, opts);
res.info(1).precision = precision;
save(save_file_name, '-struct', 'res');

% for plotting
iter = 2;
done = false;
while ~done
  fprintf('iter %d: querying random examples\n', iter);
  new_vetted_set = random_sampling(prob, res.info(iter-1).to_use_vetted, vetted_labels, opts);
  res.info(iter).to_use_vetted = new_vetted_set; % vetting
  res.info(iter).current_budget = res.info(iter-1).current_budget + batch_budget;
  assert(sum(sum(new_vetted_set))== res.info(iter).current_budget);

  % use this to decouple the effect of the sample vs estimator
  %precision = get_precision(estimator, new_vetted_set,...
  %  label_set, prob, opts);

  % retrain the model
  fprintf('iter %d: retraining estimator\n', iter);
  estimator = train_cnn(imdb, label_set,...
    res.info(iter).to_use_vetted, resdb, opts);

  precision = get_precision(estimator, res.info(iter).to_use_vetted,...
    label_set, prob, opts);
  res.info(iter).precision = precision;

  save(save_file_name, '-struct', 'res');
  if res.info(iter).current_budget >  total_budget - batch_budget
    fprintf('Not enough budget left, exiting...\n')
    done = true;
  end
  iter = iter +1;
end

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
scores = prob(:, all_videos);
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
exp_name = sprintf('sm_random_%d', current_budget);
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
  fc1000 = resdb.fc1000.outputs(new_imdb.tags_to_train, :);
  fc1000_path = fullfile(path_dir, 'fc1000.mat');
  save(fc1000_path, 'fc1000');
  fprintf('Done\n');

  % run the learner
  [net,~]=run_train_language(exp_name, opts.gpu);
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
