function run_sm_at_random_nuswide(imdb, resdb, vetted_labels, varargin)
% run multi-model active testing basic version
% Args:
%   imdb: the common imdb
%   model_A: struct, have the vetted indeces + estimator_A
%   resdb_A: the resdb for A
%   resdb_A: the resdb for B
%   vetted_labels: the whole vetted_labels
opts.max_budget = 48*81;
opts.batch_budget = 4*81;
opts.k_evaluation = 48;
opts.search_mode = 'global';
opts.estimator = 'learner';
opts.dataset = 'nuswide';
opts.strategy = 'random';
opts.lambdas = [0.3 0.3];
opts.cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
opts.gpu = 1;
opts.seed = 1;
opts = vl_argparse(opts, varargin);

% fprintf('strategy: %s, estimator: %s, dataset: %s\n',...
%   opts.strategy, opts.estimator, opts.dataset);

opts.distance = load('active_testing/nuswide_vetted_distance.mat');

% need to change this
fid = fopen('active_testing/nuswide/Subset81.txt');
total_tags = numel(imdb.classes.name);
tags = textscan(fid, '%s\r\n'); tags = tags{1}; % this is another difference
tag_indeces_bin = false(total_tags, 1);
for index = 1:numel(tags)
  tag_index = strcmp(imdb.classes.name, tags{index});
  if tag_index == 0
    fprintf('can not find %s\n', tags{index});
  end
  tag_indeces_bin(tag_index) = true;
end
opts.tag_indeces = find(tag_indeces_bin);
opts.imdb = imdb;
opts.resdb_videoids = resdb.video_ids;

% vetted labels
% need to change this
label_set.vetted_labels = imdb.images.vetted_labels(opts.tag_indeces, resdb.video_ids);
label_set.observed_label = imdb.images.label(opts.tag_indeces, resdb.video_ids);

% load the classifier scores
prob = resdb.fc1000.outputs(opts.tag_indeces, :);

total_budget = opts.max_budget;
batch_budget = opts.batch_budget;
save_file_name = sprintf('active_testing/res_sm_%s_%s_%s.mat', opts.strategy,...
  opts.estimator, opts.dataset);
rng(opts.seed);
res = struct();
res.name = sprintf('%s-%s', opts.strategy, opts.estimator);
vetted_examples = random_sampling(prob, zeros(size(label_set.vetted_labels)),...
  label_set.vetted_labels, opts);
res.info(1).to_use_vetted = sparse(vetted_examples); % the visible part of the vetted matrix
res.info(1).current_budget = sum(sum(vetted_examples));

opts = update_params(label_set, res.info(1).to_use_vetted, resdb, prob, opts);
precision = get_precision_all(res.info(1).to_use_vetted, label_set, prob, opts);
res.info(1).precision = precision;
save(save_file_name, '-struct', 'res');
return

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
      new_vetted_set = most_confused(prob, res.info(iter-1).to_use_vetted, vetted_labels, opts);
    otherwise
  end
  res.info(iter).to_use_vetted = new_vetted_set; % vetting
  res.info(iter).current_budget = full(sum(sum(res.info(iter).to_use_vetted)));
  % use this to decouple the effect of the sample vs estimator
  precision_before_retrained = get_precision_all(res.info(iter).to_use_vetted, label_set, prob, opts);
  res.info(iter).precision_before_retrained = precision_before_retrained;
  % retrain the model
  fprintf('iter %d: updating the priors\n', iter);
  opts = update_params(label_set, res.info(iter).to_use_vetted, resdb, prob, opts);
  precision = get_precision_all(res.info(iter).to_use_vetted, label_set, prob, opts);
  res.info(iter).precision = precision;
  
  fprintf('iter %d: prec: %0.2f cur_budget: %d total %d batch %d\n', iter,...
    mean(precision)*100,...
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
  case 'prior_sigmoid'
    opts.priors = update_prior_sigmoid(to_use_vetted, label_set, prob, opts);
  case 'prior3'
    opts.priors = update_prior3(to_use_vetted, label_set, prob, opts);
  case 'prior_nb'
    opts.priors = update_prior_nb(to_use_vetted, label_set, prob, opts);
  case 'lasso'
    opts.priors = update_lasso(to_use_vetted, label_set, prob, opts);
  case 'naive'
    return
  case 'nonparam'
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
  case 'prior_sigmoid'
    priors = opts.priors;
    precisions = get_precision_prior_sigmoid(to_use_vetted,...
     label_set, prob, priors, opts);
  case 'prior3'
    priors = opts.priors;
    precisions = get_precision_prior3_based(to_use_vetted,...
      label_set, prob, priors, opts);
  case 'prior_nb'
    priors = opts.priors;
    precisions = get_precision_prior_nb(to_use_vetted,...
      label_set, prob, priors, opts);
  case 'lasso'
    priors = opts.priors;
    precisions = get_precision_lasso(to_use_vetted,...
      label_set, prob, priors, opts);
  case 'nonparam'
    precisions = get_precision_nonparam(to_use_vetted,...
      label_set, prob, opts);
  case 'naive'
    precisions = get_precision_naive(to_use_vetted, label_set, prob, opts);
  otherwise
    error('Unrecognized estimator');
end

% -------------------------------------------------------------------------
function priors = update_prior(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

num_tag = numel(tag_indeces);
% computing the the basic base
base_y0z1 = 0; base_y0 = 0;
base_y1z1 = 0; base_y1 = 0;
for index = 1:num_tag
  tag_index = index;
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index, :));
  [topk_vetted_videoid, ~, ~] = intersect(vetted_videoid, topk_videoid);
  y = observed_label(tag_index, topk_vetted_videoid);
  z = vetted_labels(tag_index, topk_vetted_videoid);
  base_y0z1 = base_y0z1 + sum(y==0 & z==1);
  base_y0 = base_y0 + sum(y==0);
  base_y1z1 = base_y1z1 + sum(y==1 & z==1);
  base_y1 = base_y1 + sum(y==1);
end
base_q = base_y0z1/base_y0;
base_p = base_y1z1/base_y1;

% find the examples and computing the precision
priors = struct;
priors.flip_priors = zeros(num_tag, 2); % p(y|z)
k = 0.1;
for index = 1:num_tag
  tag_index = index;
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index, :));
  [topk_vetted_videoid,~,~] = intersect(vetted_videoid, topk_videoid);
%   topk_vetted_videoid = topk_videoid;
  y = observed_label(tag_index, topk_vetted_videoid);
  z = vetted_labels(tag_index, topk_vetted_videoid);
  priors.flip_priors(index, 1) = (sum(y==0 & z==1)+0.1)/(sum(y==0)+0.2);
  priors.flip_priors(index, 2) = (sum(y==1&z==1)+0.1)/(sum(y==1)+0.2);
%   priors.flip_priors(index, 1) = (sum(y==0 & z==1)+k*base_q)/(sum(y==0)+k);
%   priors.flip_priors(index, 2) = (sum(y==1&z==1)+k*base_p)/(sum(y==1)+k);
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior_based(to_use_vetted, label_set,...
  prob, priors, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
observed_label = label_set.observed_label;
vetted_labels = label_set.vetted_labels;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index = index;
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prior = observed_label(tag_index, unvetted_videoid) * priors.flip_priors(index, 2);
  neg_prior = ~observed_label(tag_index, unvetted_videoid) * priors.flip_priors(index, 1);
  unvetted_count = full(sum(pos_prior) + sum(neg_prior));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior_sigmoid(to_use_vetted, label_set,...
  prob, priors, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  s = prob(tag_index, unvetted_videoid);
  q = priors.q(index);
  p = priors.p(index);
  
  A = priors.A_and_B{index}(1); B = priors.A_and_B{index}(2);
  r = 1./(1+exp(A*s+B));
  pos = full(observed_label(tag_index, unvetted_videoid));
  pos_prior = pos .* (p*r+eps)./(p*r+q*(1-r)+eps); % y = 1
  
  neg = full(~observed_label(tag_index, unvetted_videoid));
  neg_prior = neg .* ((1-p)*r+eps)./((1-q)*(1-r) + (1-p)*r + eps);
  unvetted_count = full(sum(pos_prior) + sum(neg_prior));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_prior_nb(to_use_vetted, label_set,...
  prob, priors, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = single(label_set.vetted_labels);
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  preds = prob(index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(index, vetted_videoid));

  % compute the unvetted precision
  q = priors.q{index};
  p = priors.p{index};
  
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  z = full(vetted_labels(index, unvetted_videoid));
  y_all = full(observed_label(:, unvetted_videoid));
  nb_val_z1 = zeros(numel(unvetted_videoid), 1);
  nb_val_z0 = zeros(numel(unvetted_videoid), 1);
  for i_unvetted = 1:numel(unvetted_videoid)
    y = y_all(:, i_unvetted);
    nb_val_z1(i_unvetted) = nb(y,1,p,q);
    nb_val_z0(i_unvetted) = nb(y,0,p,q);
  end
  
%   A = priors.A_and_B{index}(1); B = priors.A_and_B{index}(2);
%   s = prob(index, unvetted_videoid);
%   z = 1./(1+exp(A*s+B))';
%   r = 0.5*ones(numel(unvetted_videoid), 1);
  z = priors.z{index};
  
  final_prob =  (nb_val_z1.*z+eps)./(nb_val_z1.*z + nb_val_z0.*(1-z)+eps);
  unvetted_count = sum(final_prob);
  
%   neg = full(~observed_label(tag_index, unvetted_videoid));
%   neg_prior = neg .* ((1-p)*r+eps)./((1-q)*(1-r) + (1-p)*r + eps);
%   unvetted_count = full(sum(pos_prior) + sum(neg_prior));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_nonparam(to_use_vetted, label_set,...
  prob, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = single(label_set.vetted_labels);
observed_label = label_set.observed_label;
distance = opts.distance;
imdb = opts.imdb;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  preds = prob(index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  [~,vetted_distance_indeces,~] = intersect(distance.videoids, vetted_videoid);
  z_vetted = vetted_labels(index, vetted_videoid);
  vetted_count = sum(z_vetted);
%   for i=1:numel(vetted_videoid)
%     subplot(numel(vetted_videoid)/2,2,i);
%     lookup_index = opts.resdb_videoids(vetted_videoid(i));
%     view_image(imdb.images.name{lookup_index}, imdb);
%     ylabel(sprintf('%d', z_vetted(i)));
%   end
  
  unvetted_videoid_all = setdiff(topK_videoid, vetted_videoid);
  z_unvetted = full(vetted_labels(index, unvetted_videoid_all));
  unvetted_count = numel(numel(unvetted_videoid_all), 1);
  k_neighbor = 1;
  for i_unvet = 1:numel(unvetted_videoid_all)
    unvetted_videoid =  unvetted_videoid_all(i_unvet);
    distance_index = find(distance.videoids==unvetted_videoid);
    d = distance.distance(distance_index, vetted_distance_indeces);
    [sorted_distance, distance_order] = sort(d);
    
%     vidid = opts.resdb_videoids(unvetted_videoid);
%     subplot(1,3,1); view_image(imdb.images.name{vidid}, imdb);
%     subplot(1,3,1); ylabel(z_unvetted(i_unvet));
%     
%     % find the nearest neighbors
%     vidid = opts.resdb_videoids(vetted_videoid(distance_order(1)));
%     subplot(1,3,2);
%     view_image(imdb.images.name{vidid}, imdb);
%     ylabel(z_vetted(distance_order(1)));
%     title(sprintf('%0.2f', sorted_distance(1)));
%     subplot(1,3,3);
%     vidid = opts.resdb_videoids(vetted_videoid(distance_order(2)));
%     view_image(imdb.images.name{vidid}, imdb);
%     ylabel(z_vetted(distance_order(2)));
%     title(sprintf('%0.2f', sorted_distance(2)));
%     unvetted_count(i_unvet) = mean(z_vetted(distance_order(1:k_neighbor)));
  end
  unvetted_count = sum(unvetted_count);
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function v = nb(y,z_i,p,q)
% -------------------------------------------------------------------------
% y - L-length vector, z_i - observed
% p - vector p(y=1|z=1), q - vector p(y=1|z=0)
if z_i == 1
  v = y.*p + (~y).*(1-p);
else
  v = y.*q + (~y).*(1-q);
end
v = prod(v);

% -------------------------------------------------------------------------
function to_use_vetted=random_sampling(prob, to_use_vetted, vetted_labels, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
num_tag = numel(tag_indeces);
for index = 1:num_tag
  already_vetted = to_use_vetted(index, :);
  condition = ~already_vetted;

  cat_prob = prob(index, :);
  [~, order] = sort(cat_prob, 'descend');
  topk_videoid = order(1:opts.k_evaluation);
  videoids_fit_condition = find(condition);
  
  vetted_shortlist = intersect(topk_videoid, videoids_fit_condition);
  random_sample = randperm(numel(vetted_shortlist));
  random_sample = random_sample(min(1:opts.batch_budget/num_tag,...
    numel(random_sample)));
  vetted_shortlist = vetted_shortlist(random_sample);

  to_use_vetted(index, vetted_shortlist) = 1;
end

% -------------------------------------------------------------------------
function priors = update_prior_sigmoid(to_use_vetted, label_set, prob, opts)
% update the prior with naive bayes estimators
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
priors = struct;
priors.p = zeros(num_tag, 1);
priors.q = zeros(num_tag, 1);
priors.A_and_B = cell(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index, :));
  topk_vetted_videoid = intersect(vetted_videoid, topk_videoid);
  y = observed_label(tag_index, topk_vetted_videoid);
  s = prob(tag_index, topk_vetted_videoid);
  z = vetted_labels(tag_index, topk_vetted_videoid);
  
  prior0 = sum(~z); prior1 = sum(z);
  if prior0 == 0 % no negative
    A = 0; B = -inf;
  elseif prior1 == 0 % no positive
    A = 0; B = inf;
  else
    [A,B] = platt(s, double(z)*2-1, prior0, prior1);
  end
  priors.A_and_B{index} = [A,B];
  
  % p = P(y=1|z=1)
  priors.p(index) = (sum(y==1 & z==1)+0.1)/(sum(z==1)+0.1);
  priors.q(index) = (sum(y==1 & z==0)+0.1)/(sum(y==0)+0.1);
end

% -------------------------------------------------------------------------
function priors = update_prior_nb(to_use_vetted, label_set, prob, opts)
% update the prior with naive bayes estimators
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
priors = struct;
priors.p = cell(num_tag, 1);
priors.q = cell(num_tag, 1);
priors.A_and_B = cell(num_tag, 1);
for index = 1:num_tag
%   tag_index = tag_indeces(index);
  preds = prob(index, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(index, :));
  topk_vetted_videoid = intersect(vetted_videoid, topk_videoid);
  
  topk_y = observed_label(:, order(1:1080));
  topk_yi = observed_label(index, order(1:1080));
  py1 = mean(topk_yi);
  py0 = mean(~topk_yi);
  
  topk_yi = repmat(topk_yi, [num_tag, 1]);
  N_y1yi1 = sum(topk_y==1 & topk_yi==1, 2);
  N_yi1 = sum(topk_yi==1, 2);
  N_y1yi0 = sum(topk_y==1 & topk_yi==0, 2);
  N_yi0 = sum(topk_yi==0, 2);
  pyyi = (N_y1yi1+0.005)./(N_yi1+0.1);
  qyyi = (N_y1yi0+0.005)./(N_yi0+0.1);
  
%   pyyi(index) = py1;
%   qyyi(index) = py1;
  
%   topk_vetted_videoid = topk_videoid;
  y = observed_label(:, topk_vetted_videoid);
  s = prob(index, topk_vetted_videoid);
  z = vetted_labels(index, topk_vetted_videoid);
  
  priors.z{index} = sum(z)/numel(z);
  priors.y{index} = mean(y(index, :));
  
  prior0 = sum(~z); prior1 = mean(z);
%   if prior0 == 0 % all positive
%     num_vetted = numel(z);
%     video_to_add = order(60:120);
%     y_topk_unvetted = full(observed_label(index, video_to_add));
%     s_topk_unvetted = full(prob(index, video_to_add));
%     [lowest_scores, ~] = sort(s_topk_unvetted(~y_topk_unvetted));
%     target = [z zeros(1, num_vetted)]*2-1;
%     scores = [s lowest_scores(1:num_vetted)];
%     [A,B] = platt(sco res, target, num_vetted, num_vetted);
%   elseif prior1 == 0 % no positive
%     num_vetted = numel(z);
%     video_to_add = order(1:120);
%     y_topk_unvetted = full(observed_label(index, video_to_add));
%     s_topk_unvetted = full(prob(index, video_to_add));
%     [highest, ~] = sort(s_topk_unvetted(~y_topk_unvetted), 'descend');
%     target = [z ones(1, num_vetted)]*2-1;
%     scores = [s highest(1:num_vetted)];
%     [A,B] = platt(scores, target, num_vetted, num_vetted);
%   else
%     [A,B] = platt(s, double(z)*2-1, prior0, prior1);
%   end
%   priors.A_and_B{index} = [A,B];
  if prior0 == 0 % no negative
    A = 0; B = -inf;
  elseif prior1 == 0 % no positive
    A = 0; B = inf;
  else
    [A,B] = platt(s, double(z)*2-1, prior0, prior1);
  end
  priors.A_and_B{index} = [A,B];
  
  % p = P(y=1|z=1)
  z = repmat(z, [num_tag, 1]);
  priors.N_y1z1{index} = sum(y==1 & z==1, 2);
  priors.N_z1{index} = sum(z==1,2);
  priors.N_y1z0{index} = sum(y==1 & z==0, 2);
  priors.N_z0{index} = sum(z==0, 2);
  
  k = 1;
  p = (priors.N_y1z1{index}+k*pyyi)./(priors.N_z1{index}+k);
  q = (priors.N_y1z0{index}+k*qyyi)./(priors.N_z0{index}+k);
%   p = (priors.N_y1z1{index}+k*pyyi)./(priors.N_z1{index}+k);
%   q = (priors.N_y1z0{index}+k*qyyi)./(priors.N_z0{index}+k);
%   lambda1 = opts.lambdas(1); lambda2 = opts.lambdas(2);
%   lambda3 = 1 - (lambda1+lambda2);

  lambda1 = 0.8;
  lambda2 = 0.2;
  
%   priors.p{index} = lambda1*p;% + lambda2*pyyi;% + lambda3*py;
  
  priors.p{index} = lambda1*p + lambda2*pyyi;% + lambda3*py;
  priors.q{index} = lambda1*q + lambda2*qyyi;% + lambda3*py;
%   priors.p{index}(index) = p(index);
%   priors.q{index}(index) = q(index);
  priors.p{index}(index) = (.9)*p(index) + .1*py1;
  %priors.q{index}(index) = (1-lambda)*q(index) + lambda*py;
end

% -------------------------------------------------------------------------
function precisions = get_precision_lasso(to_use_vetted, label_set,...
  prob, priors, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = single(label_set.vetted_labels) .* full(to_use_vetted); % 1Kx160K
observed_label = label_set.observed_label;
betas = priors.beta;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  y = full(observed_label(:, unvetted_videoid));
  s = full(prob(:, unvetted_videoid));
  input = [y; s];
  %input = [y];
  fit = priors.beta{index};
  if numel(unvetted_videoid) > 0
    z = glmnetPredict(fit, input', [], 'response');
    z = sum(z, 1);
    z = z(min(15, numel(z)));
  else
    z = 0;
  end
  
%   z = sum(y'*beta(:, 1));
  
  precisions(index) = (vetted_count + z)/opts.k_evaluation;
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
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
all_videos = cell(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
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
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  [~,~,ib] = intersect(unvetted_videoid, all_videos);
  pos_prob = estimator_scores(tag_index, ib);
  unvetted_count = sum(pos_prob);
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function precisions = get_precision_naive(to_use_vetted, label_set,...
  prob, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index = index;
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_videoid = find(to_use_vetted(tag_index, :));
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index, vetted_videoid));

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prob = observed_label(tag_index, unvetted_videoid);
  unvetted_count = full(sum(pos_prob));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
end

% -------------------------------------------------------------------------
function to_use_vetted=mcn(prob, to_use_vetted, label_set, opts)
% -------------------------------------------------------------------------
observed_label = label_set.observed_label;
vetted_labels = label_set.vetted_labels;
tag_indeces = opts.tag_indeces;
num_tag = numel(tag_indeces);
lookup_indeces = cell(num_tag, 1);
scores_all = cell(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);

  cat_prob = prob(tag_index, :);
  [shortlist_scores, shortlist_order] = sort(cat_prob, 'descend');
  shortlist_scores = shortlist_scores(1:opts.k_evaluation);
  topk_videoids = shortlist_order(1:opts.k_evaluation);

  negative = ~observed_label(tag_index, :);
  not_already_vetted = ~to_use_vetted(tag_index, :);

  condition = (negative & not_already_vetted);
  [~, ~, ib] = intersect(find(condition), topk_videoids);
  videoids_fit_scores = shortlist_scores(ib);
  scores_all{index} = videoids_fit_scores;

  indeces = [tag_index * ones(numel(videoids_fit_scores), 1) shortlist_order(ib)'];
  lookup_indeces{index} = indeces;
end
scores_all = cat(2, scores_all{:});
lookup_indeces = cat(1, lookup_indeces{:});

[~, order] = sort(scores_all, 'descend');
to_take = min(opts.batch_budget, numel(order));
indeces = lookup_indeces(order(1:to_take), :);
sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
to_use_vetted(sub) = 1;

if to_take < opts.batch_budget
  lookup_indeces = cell(num_tag, 1);
  scores_all = cell(num_tag, 1);
  for index = 1:num_tag
    tag_index = tag_indeces(index);

    cat_prob = prob(tag_index, :);
    [shortlist_scores, shortlist_order] = sort(cat_prob, 'descend');
    shortlist_scores = shortlist_scores(1:opts.k_evaluation);
    topk_videoids = shortlist_order(1:opts.k_evaluation);

    positive = observed_label(tag_index, :);
    not_already_vetted = ~to_use_vetted(tag_index, :);

    condition = (positive & not_already_vetted);
    [~, ~, ib] = intersect(find(condition), topk_videoids);
    videoids_fit_scores = shortlist_scores(ib);
    scores_all{index} = videoids_fit_scores;

    indeces = [tag_index * ones(numel(videoids_fit_scores), 1) shortlist_order(ib)'];
    lookup_indeces{index} = indeces;
  end
  scores_all = cat(2, scores_all{:});
  lookup_indeces = cat(1, lookup_indeces{:});
  
  [~, order] = sort(scores_all);
  to_take2 = opts.batch_budget - to_take;
  to_take2 = min(order, to_take2);
  indeces = lookup_indeces(order(1:to_take2), :);
  sub = sub2ind(size(to_use_vetted), indeces(:, 1), indeces(:, 2));
  to_use_vetted(sub) = 1;
end

% -------------------------------------------------------------------------
function priors = update_lasso(to_use_vetted, label_set, prob, opts)
% -------------------------------------------------------------------------
tag_indeces = opts.tag_indeces;
vetted_labels = label_set.vetted_labels; % 1Kx160K
% visible_vetted_labels = single(vetted_labels) .* single(full(to_use_vetted));
observed_label = label_set.observed_label;

% get all the videos in the top-K
num_tag = numel(tag_indeces);
% find the examples and computing the precision
priors = struct;
priors.beta = cell(num_tag, 1);
for index = 1:num_tag
  tag_index = tag_indeces(index);
  preds = prob(tag_index, :);
  [~, order] = sort(preds, 'descend');
  
  topk_videoid = order(1:opts.k_evaluation);
  vetted_videoid = find(to_use_vetted(tag_index, :));
  topk_vetted_videoid = intersect(vetted_videoid, topk_videoid);
  
  sure_negative = order(200:200+numel(vetted_videoid));
  y_sure_negative = full(observed_label(:, sure_negative));
  s_sure_negative = full(prob(:, sure_negative));
  z_sure_negative = zeros(1, numel(sure_negative));
  
  y = full(observed_label(:, topk_vetted_videoid));
  s = prob(:, topk_vetted_videoid);
  %inputs = [y' ones(numel(y), 1)];
%   inputs = [y' ones(numel(y), 1)];
  z = single(full(vetted_labels(tag_index, topk_vetted_videoid)));
  unique_z = unique(z);
  if numel(unique_z) == 2
    y2 = y;
    s2 = s;
    z2 = z;
  else
    if unique_z == 1
      y2 = y;
      s2 = s;
      z2 = z;
      z2(randi(numel(z2))) = 0;
    else
      y2 = y;
      s2 = s;
      z2 = z;
      z2(randi(numel(z2))) = 1;
    end
  end
  input = [y2; s2];
  %input = [y2];
  priors.beta{index} = glmnet(input', z2'+1, 'binomial');
end