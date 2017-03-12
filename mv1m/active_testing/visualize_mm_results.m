% this script visualize the mm results
result_basic = load('active_testing/mm_res_adaptive_global.mat');
result_dual = load('active_testing/mm_dual_res_adaptive_global.mat');
result_joint = load('active_testing/mm_joint_res_adaptive_global.mat');
true_precision_A = load('active_testing/true_precision_aria.mat');
true_precision_A = true_precision_A.precisions_A;
true_precision_B = load('active_testing/true_precision_aria_ub.mat');
true_precision_B = true_precision_B.precisions_B;
figure_storage = '/home/phuc/Research/iccv2017/figures';

%% basic
num_point = numel(result_basic.info);
x = zeros(num_point, 1);
y = zeros(num_point, 2);
for i=1:numel(result_basic.info)
  num_vetted = full(sum(sum(result_basic.info(i).to_use_vetted)));
  pA = mean(result_basic.info(i).precision_A);
  pB = mean(result_basic.info(i).precision_B);
  x(i) = num_vetted;
  y(i, :) = [pA pB];
  fprintf('nv %d pA %0.4f (%0.4f) pB %0.4f (%0.4f)\n', num_vetted,...
    pA, mean(true_precision_A),...
    pB, mean(true_precision_B));
end

close all;
plot(x, y, 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
plot(x, ones(numel(x), 1)*mean(true_precision_B), 'LineWidth', 2);
grid on;

legend({'pA', 'pB', 'p*A', 'p*B'});
xlabel('#vetted');
ylabel('prec@48');
figure_name = fullfile(figure_storage, 'mm_at_basic');
export_fig(figure_name, '-pdf', '-transparent');
% export_fig(figure_name, '-png', '-transparent');

%% dual
num_point = numel(result_dual.info);
x_dual = zeros(num_point, 1);
y_dual = zeros(num_point, 2);
for i=1:numel(result_dual.info)
  num_vetted = full(sum(sum(result_dual.info(i).to_use_vetted)));
  pA = mean(result_dual.info(i).precision_A);
  pB = mean(result_dual.info(i).precision_B);
  x(i) = num_vetted;
  y(i, :) = [pA pB];
  fprintf('nv %d pA %0.4f (%0.4f) pB %0.4f (%0.4f)\n', num_vetted,...
    pA, mean(true_precision_A),...
    pB, mean(true_precision_B));
end

close all;
plot(x, y, 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
plot(x, ones(numel(x), 1)*mean(true_precision_B), 'LineWidth', 2);
grid on;

% legend({'pA', 'pB', 'p*A', 'p*B'});
% xlabel('#vetted');
% ylabel('prec@48');
% figure_name = fullfile(figure_storage, 'mm_at_dual');
% export_fig(figure_name, 'm2', '-pdf', '-transparent');
% export_fig(figure_name, '-m2', '-png', '-transparent');

%% joint
num_point = 20;%numel(result_dual.info);
x = zeros(num_point, 1);
y_joint = zeros(num_point, 2);
for i=1:20%numel(result_joint.info)
  num_vetted = full(sum(sum(result_joint.info(i).to_use_vetted)));
  pA = mean(result_joint.info(i).precision_A);
  pB = mean(result_joint.info(i).precision_B);
  x(i) = num_vetted;
  y_joint(i, :) = [pA pB];
  fprintf('nv %d pA %0.4f (%0.4f) pB %0.4f (%0.4f)\n', num_vetted,...
    pA, mean(true_precision_A),...
    pB, mean(true_precision_B));
end

plot(x, y_joint, '--', 'LineWidth', 2);
hold on;
plot(x, ones(numel(x), 1)*mean(true_precision_A), 'LineWidth', 2);
plot(x, ones(numel(x), 1)*mean(true_precision_B), 'LineWidth', 2);
grid on;

%%
figure_name = fullfile(figure_storage, 'mm_at_dual_vs_joint');
export_fig(figure_name, '-pdf', '-transparent');
export_fig(figure_name, '-pdf', '-transparent');