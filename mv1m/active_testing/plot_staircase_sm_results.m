function plot_staircase_sm_results(all_results, true_precision)
num_tag = 75;
total_budget = num_tag * 48;
num_point = min(11, numel(all_results{1}.info));
x = zeros(num_point+1, 1);
d = zeros(num_point+1, numel(all_results));
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
    d_mcn_before = mean(abs(res.info(i).precision_before_retrained - true_precision));
    d_mcn = mean(abs(res.info(i).precision - true_precision));
    v_gain = [v_gain (d_mcn_before - mean(abs(res.info(i-1).precision - true_precision)))];
    e_gain = [e_gain (d_mcn - d_mcn_before)];
    d = [d d_mcn_before d_mcn];
  end
  
  figure(2);
  plot(x*1000/3600, d, 'o-', 'LineWidth', 2);
  hold on;
  grid on
  legends{j} = res.name;
end
legend(legends);
xlabel('%s vetted');
ylabel('delta(prec@48)');