function compose_csv(datasets, main_dataset)
% compose_csv({'ari_full', 'ari_small'}, main_dataset)
res_name = sprintf('%s_results.mat', strjoin(datasets, '_'));
save_path = fullfile('/home/phuc/Research/danilo/mv1m/', res_name);
if ~exist(save_path, 'file')
  aggregate_results = struct();
  aggregate_results.datasets = datasets;
  aggregate_results.main_dataset = main_dataset;
  exp_dir_root = '/mnt/large/pxnguyen/cnn_exp/';
  for dataset_index = 1:numel(datasets)
    dataset_name = datasets{dataset_index};
    exp_dir = fullfile(exp_dir_root, datasets{dataset_index});
    imdb = load(fullfile(exp_dir, sprintf('%s_imdb.mat', dataset_name)));
    info = load(fullfile(exp_dir, 'info.mat'));

    train = (imdb.images.set == 1);
    vid_count = full(sum(imdb.images.label(:, train), 2));

    aggregate_results.(dataset_name).tag_names = imdb.classes.name;
    aggregate_results.(dataset_name).vid_count = info.train_vid_count;
    aggregate_results.(dataset_name).AP = info.AP_tag;
  end

  % computing the chance AP
  eval_indeces = (imdb.images.set == 2);
  N = sum(eval_indeces);
  aggregate_results.chance_AP = zeros(numel(imdb.classes.name), 1);
  fprintf('Computing the chance AP...\n');
  for index = 1:length(imdb.classes.name)
    index
    class_indeces = imdb.images.label(index, :);
    eval_vid_indeces = find(class_indeces & eval_indeces);
    num_positives = numel(eval_vid_indeces);
    aggregate_results.chance.AP(index) = num_positives;
    aggregate_results.chance.AP(index) = single(num_positives)/N;
  end
  fprintf('Saving the results...\n');
  save(save_path, '-struct', 'aggregate_results');
else
  aggregate_results = load(save_path);
end
print_to_csv(aggregate_results);

function print_to_csv(aggregate_res)
fprintf('Writing to csv...\n');
fid = fopen('ap.csv', 'w');
fprintf(fid, 'tag,num_train,AP,dataset\n');
main_tags = aggregate_res.(aggregate_res.main_dataset).tag_names;
for dataset_index = 1:length(aggregate_res.datasets)
  dataset_name = aggregate_res.datasets{dataset_index};
  for tag_index = 1:numel(aggregate_res.(dataset_name).tag_names)
    tag_name = aggregate_res.(dataset_name).tag_names(tag_index);
    if find(strcmp(tag_name, main_tags))
      fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
        aggregate_res.(dataset_name).tag_names{tag_index},...
        full(aggregate_res.(dataset_name).vid_count(tag_index)),...
        aggregate_res.(dataset_name).AP(tag_index),...
        dataset_name);
    end
  end
end
fclose(fid);
