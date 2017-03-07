function multi_model_active_testing(exp_name)
[~, hostname] = system('hostname');
hostname = strtrim(hostname);
switch hostname
  case 'pi'
    cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
    vetted_labels_storage = '/home/phuc/Research/yaromil';
  case 'omega'
    cnn_exp = '/home/nguyenpx/cnn_exp/';
    vetted_labels_storage = '/home/nguyenpx/vetted_labels/';
end
fprintf('Loading imdb\n');
aria_imdb = load(fullfile(cnn_exp, 'aria/aria_imdb.mat'));

fprintf('Loading the vetted labels\n');
vetted_labels_train = load(fullfile(vetted_labels_storage, 'vetted_labels_train.mat'));
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load(fullfile(vetted_labels_storage, 'vetted_labels_test.mat'));
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

% load the resdb
fprintf('Loading classifier scores\n');
resdb_A = load(fullfile(cnn_exp, 'aria/resdb-iter-390000-test.mat'));
resdb_B = load(fullfile(cnn_exp, 'aria_upperbound/resdb-iter-172000-test.mat'));
% todo: reduce the size of resdb_B by removing the gts.

fprintf('Loading init models\n');
resA_adaptive = load('active_testing/res_adaptive.mat');
model_A = struct();
model_A.to_use_vetted = resA_adaptive.info(4).to_use_vetted;
net = load(fullfile(cnn_exp, 'adaptive_1200_global/net-epoch-3-iter-202000.mat'));
model_A.estimator = dagnn.DagNN.loadobj(net.net);
switch exp_name
  case 'basic'
    run_mm_at_basic(aria_imdb, model_A, resdb_A, resdb_B, vetted_labels)
  case 'dual'
    run_mm_at_dual(aria_imdb, model_A, resdb_A, resdb_B, vetted_labels)
  case 'cotraining'
    error('Have not been implemented\n');
  otherwise
    error('exp_name not recognize\n')
end