%% this script explores the experiment of multi-model active-testing
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');
vetted_labels_train = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load('/home/phuc/Research/yaromil/vetted_labels_test.mat');
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

%%
aria_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-390000-test.mat');
aria_ub_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_upperbound/resdb-iter-172000-test.mat');

%%
% load the tag list
fid = fopen('active_testing/tags.list');
tags = textscan(fid, '%s\n'); tags = tags{1};
lstruct = load('active_testing/true_precision_at_48.mat');
true_precisions = lstruct.prec;

%% run the basic algorithm
resdb_A = aria_resdb;
resdb_B = aria_ub_resdb;
resA_adaptive = load('active_testing/res_adaptive.mat');
model_A = struct();
model_A.to_use_vetted = resA_adaptive.info(4).to_use_vetted;
net = load('/mnt/large/pxnguyen/cnn_exp/adaptive_1200_global/net-epoch-3-iter-202000.mat');
model_A.estimator = dagnn.DagNN.loadobj(net.net);
run_mm_at_basic(aria_imdb, model_A, resdb_A, resdb_B, vetted_labels)

%% run the dual algorithm
resdb_A = aria_resdb;
resdb_B = aria_ub_resdb;
resA_adaptive = load('active_testing/res_adaptive.mat');
model_A = struct();
model_A.to_use_vetted = resA_adaptive.info(4).to_use_vetted;
net = load('/mnt/large/pxnguyen/cnn_exp/adaptive_1200_global/net-epoch-3-iter-202000.mat');
model_A.estimator = dagnn.DagNN.loadobj(net.net);
run_mm_at_dual(aria_imdb, model_A, resdb_A, resdb_B, vetted_labels)