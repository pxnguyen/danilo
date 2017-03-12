function single_model_active_testing(exp_name, resdb, gpu)
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
%fprintf('Loading classifier scores\n');
%resdb_A = load(fullfile(cnn_exp, 'aria/resdb-iter-390000-test.mat'));

fprintf('Loading init models\n');
switch exp_name
  case 'random'
    run_sm_at_random(aria_imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp, 'gpu', gpu);
  case 'mcn'
    run_sm_at_mcn(aria_imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp, 'gpu', gpu);
  case 'adaptive'
    run_mm_at_adaptive(aria_imdb, resdb, vetted_labels, 'cnn_exp', cnn_exp, 'gpu', gpu);
    error('Have not been implemented\n');
  otherwise
    error('exp_name not recognize\n')
end
