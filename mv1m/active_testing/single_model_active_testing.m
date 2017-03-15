function single_model_active_testing(strategy, estimator, resdb, gpu, varargin)
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
dataset = 'aria';
imdb = load(fullfile(cnn_exp, sprintf('%s/%s_imdb.mat', dataset, dataset)));

fprintf('Loading the vetted labels\n');
vetted_labels_train = load(fullfile(vetted_labels_storage, 'vetted_labels_train.mat'));
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load(fullfile(vetted_labels_storage, 'vetted_labels_test.mat'));
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

% load the resdb
%fprintf('Loading classifier scores\n');
%resdb_A = load(fullfile(cnn_exp, 'aria/resdb-iter-390000-test.mat'));

fprintf('Loading init models\n');
run_sm_at_random(imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp,...
  'gpu', gpu, 'strategy', strategy, 'estimator', estimator, varargin{:});
% switch strategy
%   case 'random'
%   case 'mcn'
%     run_sm_at_mcn(imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp, 'gpu', gpu, 'estimator', estimator);
%   case 'adaptive'
%     run_sm_at_adaptive(imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp, 'gpu', gpu, 'estimator', 'learner');
%   otherwise
%     error('exp_name not recognize\n')
% end