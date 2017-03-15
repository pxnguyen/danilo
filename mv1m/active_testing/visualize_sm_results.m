% visualize the single model results
true_precision_A = load('active_testing/true_precision_aria.mat');
true_precision_A = true_precision_A.precisions_A;

aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');

result_random_learner = load('active_testing/res_sm_random_learner_global.mat');
result_random_learner.name = 'random-learner';
result_random_naive = load('active_testing/res_sm_random_naive_global.mat');
result_random_naive.name = 'random-naive';
result_mcn_naive = load('active_testing/res_sm_mcn_naive_global.mat');
result_mcn_naive.name = 'mcn-naive';
result_mcn_learner = load('active_testing/res_sm_mcn_learner_global.mat');
result_mcn_learner.name = 'mcn-learner';
result_random_prior = load('active_testing/res_sm_random_prior_global.mat');
result_random_prior2 = load('active_testing/res_sm_random_prior2_global.mat');
result_random_prior2.name = 'random-prior2';
result_random_prior3 = load('active_testing/res_sm_random_prior3_global.mat');
result_random_prior3.name = 'random-prior3';
result_mcn_prior = load('active_testing/res_sm_mcn_prior_global.mat');
result_mcn_prior.name = 'mcn-prior';
result_adaptive_prior = load('active_testing/res_sm_adaptive_prior_global.mat');
result_adaptive = load('active_testing/res_sm_adaptive_learner_global.mat');
figure_storage = '/home/phuc/Research/iccv2017/figures';
% result_adaptive = load('active_testing/mm_joint_res_adaptive_global.mat');

%%
all_results = {...
  result_random_prior,...
  result_adaptive_prior,...
};

num_point = 6;
x = zeros(num_point, 1);
y = zeros(num_point, numel(all_results));
d = zeros(num_point, numel(all_results));
for i=1:num_point%numel(result_random_naive.info)
  i
  num_vetted = full(sum(sum(all_results{1}.info(i).to_use_vetted)));
  x(i) = num_vetted;
  for j=1:numel(all_results)
    res = all_results{j};
    res.name
    y(i, j) = mean(res.info(i).precision);
    d(i, j) = mean(abs(res.info(i).precision - true_precision_A));
  end
end

legends = cell(numel(all_results), 1);
for i=1:numel(all_results)
  legends{i} = all_results{i}.name;
end

close all;
plot(x, y, '-o', 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
grid on;
xlabel('#vetted');
ylabel('prec@48');
legend(legends);

figure(2);
plot(x, d, '-o', 'LineWidth', 2);
grid on
xlabel('#vetted');
ylabel('delta prec@48');
legend(legends);

%%
all_results = {...
  result_random_prior,...
  result_random_prior2,...
  result_random_learner,...
  result_mcn_naive,...
};

num_point = 8;
close all;
for j=1:numel(all_results)
  figure(j);
  res = all_results{j};
  x = zeros(num_point, 1);
%   y = zeros(75, num_point);
  d = zeros(75, num_point);
  y = zeros(75, num_point);
  for i=1:num_point%numel(result_random_naive.info)
    num_vetted = full(sum(sum(res.info(i).to_use_vetted)));
    x(i) = num_vetted;
    y(:, i) = res.info(i).precision;
    d(:, i) = abs(res.info(i).precision - true_precision_A);
  end
  subplot(2, 1, 1); plot(x,d'); title(sprintf('%s: delta', res.name));
  grid on;
  subplot(2, 1, 2); plot(x,y'); title(sprintf('%s: y', res.name));
  grid on;
end
  

%% naive
num_point = 9%numel(result_random_naive.info);
x = zeros(num_point, 1);
y = zeros(num_point, 6);
d = zeros(num_point, 6);
for i=1:num_point%numel(result_random_naive.info)
  num_vetted = full(sum(sum(result_random_naive.info(i).to_use_vetted)));
  p_random_learner = mean(result_random_learner.info(i).precision);
  p_random_naive = mean(result_random_naive.info(i).precision);
  p_random_prior = mean(result_random_prior.info(i).precision);
  p_mcn_learner = mean(result_mcn_learner.info(i).precision);
  p_mcn_naive = mean(result_mcn_naive.info(i).precision);
  p_adaptive = mean(result_adaptive.info(i).precision);
%     p_mcn = mean(result_mcn.info(i).precision);
  x(i) = num_vetted;
  y(i, :) = [p_random_naive, p_random_learner, p_mcn_naive, p_mcn_learner, p_random_prior, p_adaptive];
  d_random_naive = mean(abs(result_random_naive.info(i).precision - true_precision_A));
  d_random_learner = mean(abs(result_random_learner.info(i).precision - true_precision_A));
  d_random_prior = mean(abs(result_random_prior.info(i).precision - true_precision_A));
  d_mcn_naive = mean(abs(result_mcn_naive.info(i).precision - true_precision_A));
  d_mcn_learner = mean(abs(result_mcn_learner.info(i).precision - true_precision_A));
  d_adaptive = mean(abs(result_adaptive.info(i).precision - true_precision_A));
  d(i, :) = [d_random_naive, d_random_learner, d_mcn_naive, d_mcn_learner, d_random_prior, d_adaptive];
end

close all;
plot(x, y, '-o', 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
grid on;
xlabel('#vetted');
ylabel('prec@48');
legend({'random-naive', 'random-learner', 'mcn-naive', 'mcn-learner', 'adative', 'true-precision'});

figure(2);
plot(x, d, '-o', 'LineWidth', 2);
grid on
legend({'random-naive', 'random-learner', 'mcn-naive', 'mcn-learner', 'adaptive', 'd_random_prior'});
xlabel('#vetted');
ylabel('delta prec@48');

%% staircase
%res = result_random_learner;
%res = result_adaptive;

close all;
all_results = {...
  result_random_prior,...
  result_adaptive_prior,...
};

legends = {};
for j = 1:numel(all_results)
  res = all_results{j};
  num_point = numel(res.info);
  x = []; x2 = [];
  y = []; y2 = [];
  d = []; d2 = [];
  v_gain = [];
  e_gain = [];
  for i=2:numel(res.info)
    num_vetted = full(sum(sum(res.info(i).to_use_vetted)));
    p_mcn_before = mean(res.info(i).precision_before_retrained);
    p_mcn = mean(res.info(i).precision);
    x = [x num_vetted num_vetted];
    y = [y p_mcn_before p_mcn];
    d_mcn_before = mean(abs(res.info(i).precision_before_retrained - true_precision_A));
    d_mcn = mean(abs(res.info(i).precision - true_precision_A));
    v_gain = [v_gain (d_mcn_before - mean(abs(res.info(i-1).precision - true_precision_A)))];
    e_gain = [e_gain (d_mcn - d_mcn_before)];
    d = [d d_mcn_before d_mcn];
  end
  
  figure(2);
  plot(x, d, 'LineWidth', 2);
  hold on;
  % plot(x2, d2, 'LineWidth', 2);
  grid on
  legends{j} = res.name;
end

legend(legends);

%%
close all;
plot(x, y, 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
grid on;
xlabel('#vetted');
ylabel('prec@48');

figure(2);
plot(x, d, 'LineWidth', 2);
title(res.name);
% hold on;
% plot(x2, d2, 'LineWidth', 2);
grid on

%% staircase
res = result_random_prior3;
num_point = numel(res.info);
x = []; x2 = [];
y = []; y2 = [];
d = []; d2 = [];
v_gain = [];
e_gain = [];
for i=2:numel(res.info)
  num_vetted = full(sum(res.info(i).to_use_vetted, 2));
  bar(histc(num_vetted, 75))
end

close all;
plot(x, y, 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
grid on;
xlabel('#vetted');
ylabel('prec@48');

figure(2);
plot(x, d, 'LineWidth', 2);
% hold on;
% plot(x2, d2, 'LineWidth', 2);
grid on
%%
figure_name = fullfile(figure_storage, 'single_model');
figure_name = fullfile(figure_storage, 'single_model_absolute');
export_fig(figure_name, '-m2', '-png', '-transparent');

%%
fid = fopen('active_testing/tags.list');
tag_set_1000 = aria_imdb.classes.name(aria_imdb.selected);
tags = textscan(fid, '%s\n'); tags = tags{1};
tag_indeces_bin = false(4000, 1);
tag_indeces_b1000_bin = false(1000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(aria_imdb.classes.name, tags{index});
  tag_indeces_bin(tag_index) = true;
  
  tag_index = strcmp(tag_set_1000, tags{index});
  tag_indeces_b1000_bin(tag_index) = true;
end
tag_indeces_1000 = find(tag_indeces_b1000_bin);
tag_set_1000 = tag_set_1000;

%%
prob = resdb_A.fc1000.outputs(aria_imdb.tags_to_train, :);
prob = vl_nnsigmoid(prob);
num_tag = numel(tag_indeces_1000);
all_videos = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');
  topK_videoid = order(1:48);
  all_videos{index} = topK_videoid;
end
% all_videos = cat(2, all_videos{:}); all_videos = unique(all_videos);

%%
vetted_labels_storage = '/home/phuc/Research/yaromil';
vetted_labels_train = load(fullfile(vetted_labels_storage, 'vetted_labels_train.mat'));
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load(fullfile(vetted_labels_storage, 'vetted_labels_test.mat'));
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];
vetted_labels = vetted_labels(aria_imdb.tags_to_train, resdb_A.video_ids);
%%
observed_label = aria_imdb.images.label(aria_imdb.tags_to_train, resdb_A.video_ids);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  cat_name = tag_set_1000{tag_index_1000};
  negative = find(~observed_label(tag_index_1000, :));
  positive = find(observed_label(tag_index_1000, :));
  vetted_videos = find(abs(vetted_labels(tag_index_1000, :)) > 1);
  vetted_negative = find(vetted_labels(tag_index_1000, :) < -1);
  vetted_positive = find(vetted_labels(tag_index_1000, :) > 1);
  fprintf('%d %d %d\n', numel(vetted_videos), numel(vetted_negative), numel(vetted_positive));
  topk_video = all_videos{index};
  vetted_topk_video = intersect(topk_video, vetted_videos);
  negative_top = intersect(negative, topk_video);
  positive_top = intersect(positive, topk_video);
  vetted_negative_top = intersect(vetted_negative, topk_video);
  vetted_positive_top = intersect(vetted_positive, topk_video);
  
  prob_topk = prob(tag_index_1000, vetted_videos);
  minv = min(prob_topk);
  maxv = max(prob_topk);
  edges = minv:0.01:maxv;
  keyboard
  subplot(3,1,1); bar(edges, histc(prob(tag_index_1000, vetted_negative), edges));
  title(sprintf('negative-%s', cat_name)); grid on;
  subplot(3,1,2); bar(edges, histc(prob(tag_index_1000, vetted_positive), edges));
  title(sprintf('positive-%s', cat_name)); grid on;
  subplot(3,1,3); bar(edges, histc(prob(tag_index_1000, vetted_videos), edges));
  title(sprintf('all-%s', cat_name)); grid on;
end

%%
observed_label = aria_imdb.images.label(aria_imdb.tags_to_train, resdb_A.video_ids);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  cat_name = tag_set_1000{tag_index_1000};
  negative = find(~observed_label(tag_index_1000, :));
  positive = find(observed_label(tag_index_1000, :));
  fprintf('%d %d %d\n', numel(vetted_videos), numel(vetted_negative), numel(vetted_positive));
  topk_video = all_videos{index};
  vetted_topk_video = intersect(topk_video, vetted_videos);
  negative_top = intersect(negative, topk_video);
  positive_top = intersect(positive, topk_video);
  
  prob_topk = prob(tag_index_1000, :);
  minv = min(prob_topk);
  maxv = max(prob_topk);
  edges = minv:0.01:maxv;
  keyboard
  subplot(3,1,1); bar(edges, histc(prob(tag_index_1000, negative), edges));
  title(sprintf('negative-%s', cat_name)); grid on;
  subplot(3,1,2); bar(edges, histc(prob(tag_index_1000, positive), edges));
  title(sprintf('positive-%s', cat_name)); grid on;
  subplot(3,1,3); bar(edges, histc(prob(tag_index_1000, :), edges));
  title(sprintf('all-%s', cat_name)); grid on;
end

%% performing platt scaling
theta = cell(num_tag, 1);
parfor index = 1:num_tag
  index
  tag_index_1000 = tag_indeces_1000(index);
  cat_name = tag_set_1000{tag_index_1000};
  negative = find(~observed_label(tag_index_1000, :));
  positive = find(observed_label(tag_index_1000, :));
  prob_cat = prob(tag_index_1000, :);
  target = observed_label(tag_index_1000, :)*2-1;
  [A, B] = platt(out, target, numel(negative), numel(positive))
  theta{index} = [A,B];
end