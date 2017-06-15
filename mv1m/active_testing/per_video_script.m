% this script explores the potential for per-video vetting.
cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
vetted_labels_storage = '/home/phuc/Research/yaromil';
fprintf('Loading imdb\n');
aria_imdb = load(fullfile(cnn_exp, 'aria/aria_imdb.mat'));

fprintf('Loading the vetted labels\n');
vetted_labels_train = load(fullfile(vetted_labels_storage, 'vetted_labels_train.mat'));
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load(fullfile(vetted_labels_storage, 'vetted_labels_test.mat'));
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];
resdb_A = load(fullfile(cnn_exp, 'aria/resdb-iter-390000-test.mat'));

%% get some statistics
vetted_labels2 = vetted_labels(:, resdb_A.video_ids);
num_lbl_vetted = sum(abs(vetted_labels2)>1, 1);
[mv, mi] = max(num_lbl_vetted);

% plotting
counts = histc(num_lbl_vetted, 1:23);
bar(counts);
grid on;
xlabel('#tags vetted');
ylabel('#video');

%% choose the videos for experiments
tag_threshold = 5;
video_ids = resdb_A.video_ids(num_lbl_vetted>tag_threshold);
save('active_testing/videos.list.mat', 'video_ids');

%%
prec_w_vetted = compute_adj_prec_at_k(aria_imdb, resdb_A, vetted_labels, 'k_evaluation', 10, 'mode', 'per-video');
prec_wo_vetted = compute_adj_prec_at_k(aria_imdb, resdb_A, aria_imdb.images.label, 'k_evaluation', 10, 'mode', 'per-video');

%%
randorder = randperm(numel(prec_w_vetted));
prec_w_vetted = prec_w_vetted(randorder);
prec_wo_vetted = prec_wo_vetted(randorder);
close all;
plot(cumsum(prec_w_vetted)'./(1:numel(prec_w_vetted)))
xlabel('# vid')
ylabel('precs')
grid on;
hold on;
plot(cumsum(prec_wo_vetted)'./(1:numel(prec_w_vetted)))

%% random - naive
budget_array = 0:500:6000;
prec_per_budget = zeros(numel(budget_array), 1);
delta_random = zeros(numel(budget_array), 1);
for i_budget = 1:numel(budget_array)
  budget = budget_array(i_budget);
  precisions=run_active_testing_per_video(resdb_A, vetted_labels,...
    'estimator', 'naive', 'budget', budget, 'k_evaluation', 10);
  prec_per_budget(i_budget) = mean(precisions);
  delta_random(i_budget) = abs(mean(precisions - prec_w_vetted));
  fprintf('%d: %0.2f %0.2f\n', budget,  prec_per_budget(i_budget)*100, delta_random(i_budget)*100);
end

%% mcn - naive
budget_array = 0:500:6000;
prec_per_budget = zeros(numel(budget_array), 1);
delta_mcn = zeros(numel(budget_array), 1);
for i_budget = 1:numel(budget_array)
  budget = budget_array(i_budget);
  precisions=run_active_testing_per_video(resdb_A, vetted_labels,...
    'estimator', 'naive', 'strategy', 'mcn', 'budget', budget, 'k_evaluation', 10);
  prec_per_budget(i_budget) = mean(precisions);
  delta_mcn(i_budget) = abs(mean(precisions - prec_w_vetted));
  fprintf('%d: %0.2f %0.2f\n', budget,  prec_per_budget(i_budget)*100, delta_mcn(i_budget)*100);
end

%% random - learner2
budget_array = 0:500:6000;
prec_per_budget = zeros(numel(budget_array), 1);
delta_mcn = zeros(numel(budget_array), 1);
for i_budget = 1:numel(budget_array)
  budget = budget_array(i_budget);
  precisions=run_active_testing_per_video(resdb_A, vetted_labels,...
    'estimator', 'naive', 'strategy', 'mcn', 'budget', budget, 'k_evaluation', 10);
  prec_per_budget(i_budget) = mean(precisions);
  delta_mcn(i_budget) = abs(mean(precisions - prec_w_vetted));
  fprintf('%d: %0.2f %0.2f\n', budget,  prec_per_budget(i_budget)*100, delta_mcn(i_budget)*100);
end

%%
close all;
plot(budget_array, delta_random, 'LineWidth', 2);
hold on;
plot(budget_array, delta_mcn, 'LineWidth', 2);
grid on;
xlabel('budget');
legend({'random-naive', 'mcn-naive'});
ylabel('delta');