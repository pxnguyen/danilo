info = load('/mnt/large/pxnguyen/cnn_exp/aria/info.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');
vetted_labels_train = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load('/home/phuc/Research/yaromil/vetted_labels_test.mat');
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

%%
aria_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-390000.mat');
prob = aria_resdb.fc1000.outputs;
%prob = vl_nnsigmoid(prob);
has_prob = zeros(size(aria_imdb.images.label));
has_prob(:, aria_resdb.video_ids) = 1;

%%
vetted_counts = sum(abs(vetted_labels_test) > 1, 2);
[sorted_count, indeces] = sort(vetted_counts, 'descend');
for index = 1:200
  tag_index = indeces(index);
  tag_name = aria_imdb.classes.name{tag_index};
  sc = sorted_count(index);
  tag_prob = prob(tag_index, :);
  [~, sorted_idx] = sort(tag_prob, 'descend');
  vid_indeces = aria_resdb.video_ids(sorted_idx(1:32));
  v_labels = vetted_labels(tag_index, vid_indeces);
  fprintf('%s: vetted count %d, %d out of 32\n', tag_name, sc, sum(abs(v_labels)>1));
end

%%
fid = fopen('active_testing/tags.list');
tags = textscan(fid, '%s\n');
tags = tags{1};

vetted_counts = sum(abs(vetted_labels_test) > 1, 2);
for index = 1:numel(tags)
  tag_name = tags{index};
  tag_index = find(strcmp(aria_imdb.classes.name, tag_name));
  tag_prob = prob(tag_index, :);
  [~, sorted_idx] = sort(tag_prob, 'descend');
  vid_indeces = aria_resdb.video_ids(sorted_idx(1:64));
  v_labels = vetted_labels(tag_index, vid_indeces);
  fprintf('%s: vetted count %d, %d out of 64\n', tag_name, vetted_counts(tag_index), sum(abs(v_labels)>1));
end

%% tag_indeces within the 1000 tags
fid = fopen('active_testing/tags.list');
tags = textscan(fid, '%s\n');
tags = tags{1};
tag_indeces_1000 = false(1000, 1);
tag_names_1000 = aria_imdb.classes.name(aria_imdb.selected);
for index = 1:numel(tags)
  tag_index = strcmp(tag_names_1000, tags{index});
  tag_indeces_1000(tag_index) = true;
end

tag_indeces_1000 = find(tag_indeces_1000);

%%
tag_indeces = false(4000, 1);
for tag_index = 1:numel(tags)
  index = find(strcmp(aria_imdb.classes.name, tags{tag_index}));
  tag_indeces(index) = true;
end

true_prec = zeros(numel(tags), 1);
original_prec = zeros(numel(tags), 1);
all_indeces = cell(numel(tags), 1);
v_labels = cell(numel(tags), 1);
o_labels = cell(numel(tags), 1);
cat_indeces = cell(numel(tags), 1); % used this to later index to the right fc1000
vetted_pairs = double(abs(vetted_labels)>1);
train_test_split = sparse(vetted_pairs);
train_test_split(:, aria_imdb.images.set==1) = 0;
train_test_split = train_test_split .* has_prob;

%% here, get the vetted_gts, and the unvetted_gts
for tag_index = 1:numel(tags)
  tag_index
  tag = tags{tag_index};
  cat_index = find(strcmp(aria_imdb.classes.name, tag));
  %cat_indeces{tag_index} = repmat(cat_index, [32, 1]);
  cat_prob = prob(cat_index, :);
  [s, indeces] = sort(cat_prob, 'descend');
  vid_indeces = aria_resdb.video_ids(indeces(1:32));
  train_test_split(cat_index, vid_indeces) = 2;
  all_indeces{tag_index} = vid_indeces;
  names = aria_imdb.images.name(vid_indeces);
  v_labels{tag_index} = vetted_labels(cat_index, vid_indeces);
  o_labels{tag_index} = aria_imdb.images.label(cat_index, vid_indeces);
  true_prec(tag_index) = sum(v_labels{tag_index}==2)/32;
  original_prec(tag_index) = sum(o_labels{tag_index}>0)/32;
end

all_indeces = cat(2, all_indeces{:});
v_labels = cat(2, v_labels{:});
o_labels = cat(2, o_labels{:});
cat_indeces = cat(1, cat_indeces{:});

%% trained models
net = load('/mnt/large/pxnguyen/cnn_exp/aria_rescore/net-epoch-2-iter-148000.mat');
net_full = dagnn.DagNN.loadobj(net.net) ;
net.move('gpu');

net_budget = load('/mnt/large/pxnguyen/cnn_exp/aria_rescore_budget/net-epoch-2-iter-262000.mat');
net_budget = dagnn.DagNN.loadobj(net_budget.net) ;

net_budget = load('/mnt/large/pxnguyen/cnn_exp/aria_rescore_budget_random/net-epoch-3-iter-262000.mat');
net_budget = dagnn.DagNN.loadobj(net_budget.net) ;

%%
[labels, test_video_ids] = find(train_test_split==2);
test_video_ids = unique(test_video_ids);
a = cell(numel(test_video_ids), 1);
b = cell(numel(test_video_ids), 1);
for i=1:numel(test_video_ids)
  prob_video = prob(:, aria_resdb.video_ids==test_video_ids(i));
  y_test = aria_imdb.images.label(aria_imdb.tags_to_train, test_video_ids(i)) > 0;
  a{i} = prob_video;
  b{i} = y_test;
end
a = single(cat(2, a{:}));
a = vl_nnsigmoid(a(aria_imdb.tags_to_train, :));
b = single(full(cat(2, b{:})));
inputs = [a; b];
inputs = permute(inputs, [3 4 1 2]);
inputs = gpuArray(single(inputs));

labels = single(rand(size(a)));
labels = permute(labels, [3 4 1 2]);
labels = gpuArray(single(labels));

net_budget.vars(2).precious = true;
net_budget.eval({'input', inputs, 'labels', labels});

fc1 = gather(net.vars(2).value);
fc1 = permute(fc1, [3 4 1 2]);
fc1 = vl_nnsigmoid(fc1);

full_fc1 = zeros(size(train_test_split));
full_fc1(aria_imdb.tags_to_train, test_video_ids) = fc1;

%% train SVM
test_indeces = find(aria_imdb.images.set==2);
count = sum(abs(vetted_labels_test) > 1, 1);
all_vid_indeces = test_indeces(count > 0);
can_used_to_train = setdiff(all_vid_indeces, all_indeces);
can_used_to_test = all_indeces;
can_used_to_train = intersect(can_used_to_train, aria_resdb.video_ids);

%% input ~ fc1000 and observed label, output ~ true label
[model, mean_X, var_X] = estimator_train(prob, aria_resdb.video_ids, aria_imdb.images.label,...
  vetted_labels, train_test_split);

%% predicting
posterior_prob = estimator_predict(prob, aria_resdb.video_ids, aria_imdb.images.label,...
  vetted_labels, train_test_split, mean_X, var_X, model);

%% set this back to a matrix
indeces = find(train_test_split==2);
fix_pos = zeros(size(train_test_split));
fix_pos(indeces) = posterior_prob(:, 1);

%% calculate the expected
unvetted_gts = full(aria_imdb.images.label(tag_indeces, aria_resdb.video_ids));
vetted_gts = vetted_labels(tag_indeces, aria_resdb.video_ids);
scores = prob(tag_indeces, :);
pos_prob = fix_pos(tag_indeces, aria_resdb.video_ids);
pos_prob_2 = full_fc1(tag_indeces, aria_resdb.video_ids);

original_precision = compute_precision_at_k(scores, unvetted_gts, 'k', 32);
expected_precision = compute_expected_precision_at_k(scores, unvetted_gts, pos_prob, 'k', 32);
expected_precision2 = compute_expected_precision_at_k(scores, unvetted_gts, pos_prob_2, 'k', 32);
true_precision = compute_precision_at_k(scores, vetted_gts > 0, 'k', 32);

%% sampling - random
num_sample = size(posterior_prob, 1);
adjusted_prec = zeros(num_sample, 1);
expected_P = zeros(num_sample, 2);
vetted = false(num_sample, 1);
order = randperm(num_sample);
for i=1:num_sample
  vetted(order(i)) = true;
  vetted_part = v_labels(vetted);
  unvetted_part = o_labels(~vetted);
  P_vetted = sum(vetted_part==2);
  P_unvetted = sum(posterior_prob(~vetted, 1));
  expected_P(i, :) = [P_vetted, P_unvetted];
  adjusted_prec(i) = (P_vetted + sum(unvetted_part==1))/num_sample;
end
expected_P = sum(expected_P, 2)/num_sample;

%% sampling - uncertainty sampling
expected_P2 = zeros(num_sample, 2);
vetted = false(num_sample, 1);
[d,order2] = sort((abs(posterior_prob(:, 1)-.5)));
for i=1:num_sample
  vetted(order2(i)) = true;
  vetted_part = v_labels(vetted);
  unvetted_part = o_labels(~vetted);
  P_vetted = sum(vetted_part==2);
  P_unvetted = sum(posterior_prob(~vetted, 1));
  expected_P2(i, :) = [P_vetted, P_unvetted];
end
expected_P2 = sum(expected_P2, 2)/num_sample;

%% plotting
close all;
figure(1);
plot([0, num_sample], [mean(true_prec), mean(true_prec)]);
hold on;
plot([0, num_sample], [mean(original_prec), mean(original_prec)]);
plot(0:num_sample, [mean(original_prec); adjusted_prec]);
plot(0:num_sample, [mean(original_prec); expected_P]);
plot(0:num_sample, [mean(original_prec); expected_P2]);
grid on;

%%
figure(2);
plot(1:num_sample, abs(mean(true_prec) - adjusted_prec));
hold on;
plot(1:num_sample, abs(mean(true_prec) - expected_P));
plot(1:num_sample, abs(mean(true_prec) - expected_P2));
grid on;
legend({'adjusted', 'exp+random', 'exp+lc'});

%% estimate prob
% pz
y = aria_imdb.images.label(aria_imdb.tags_to_train, can_used_to_train);
z = vetted_labels(aria_imdb.tags_to_train, can_used_to_train);

pos_vetted_indeces = find(z==2);
p1 = sum(y(pos_vetted_indeces))/numel(pos_vetted_indeces);

neg_vetted_indeces = find(z==-2);
p0 = sum(y(neg_vetted_indeces))/numel(neg_vetted_indeces);
%% qic
% get all the top-32 list, cross check with can_used_train, check to see
% how many is 1.
num_test = sum(aria_imdb.images.set==2);
top32 = false(numel(aria_imdb.tags_to_train), num_test);
tags_to_train = find(aria_imdb.tags_to_train);
total_pos_vetted = 0;
total_vetted = 0;
for index = 1:numel(tags)
  tag_index = tags_to_train(index);
  [~, order] = sort(prob(tag_index, :), 'descend');
  top32 = aria_resdb.video_ids(order(1:32));
  vetted_top32 = vetted_labels(tag_index, top32);
  total_pos_vetted = total_pos_vetted + sum(vetted_top32==2);
  total_vetted = total_vetted + sum(abs(vetted_top32)==2);
end