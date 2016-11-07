function subsample_dataset(dataset, fraction)
% create a smaller dataset by subsampling training data
% Args:
%   dataset: the dataset name to be subsample
%   fraction: the fraction to scale down (between 0-1)
dataset_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', dataset);
new_dataset = sprintf('%s_small', dataset);
new_dataset_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', new_dataset);
if ~exist(new_dataset_dir, 'dir')
  mkdir(new_dataset_dir);
end
new_imdb_path = fullfile(new_dataset_dir, sprintf('%s_imdb.mat', new_dataset));
new_imdb = load(fullfile(dataset_dir, sprintf('%s_imdb.mat', dataset)));

train_indeces = (new_imdb.images.set==1);
tags = new_imdb.classes.name;
vid_indeces_discard = cell(numel(tags), 1);
for tag_index = 1:numel(tags)
  class_indeces = new_imdb.images.label(tag_index, :);
  train_class_indeces = find(class_indeces & train_indeces);
  random_order = randperm(numel(train_class_indeces));
  num_discard = uint32(ceil(numel(random_order)*(1-fraction)));
  fprintf('%d %s %d\n', tag_index, tags{tag_index}, num_discard);
  vid_indeces_discard{tag_index} = train_class_indeces(random_order(1:num_discard));
end

vid_indeces_discard = cat(2, vid_indeces_discard{:});
new_imdb.images.name(vid_indeces_discard) = [];
new_imdb.images.id(vid_indeces_discard) = [];
new_imdb.images.set(vid_indeces_discard) = [];
new_imdb.images.label(:, vid_indeces_discard) = [];

save(new_imdb_path, '-struct', 'new_imdb');
