function filter_spam(dataset)
% filter out the spam videos in the imdb according to
% spam_vids.list, and output a new imdb.
% sample: filter_spam('ari_full')
% Args:
%   dataset: the name of the dataset to filter out
dataset_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', dataset);
new_dataset = sprintf('%s_nospam', dataset);
new_dataset_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', new_dataset);
if ~exist(new_dataset_dir, 'dir')
  mkdir(new_dataset_dir);
end
new_imdb_path = fullfile(new_dataset_dir, sprintf('%s_imdb.mat', new_dataset));
new_imdb = load(fullfile(dataset_dir, sprintf('%s_imdb.mat', dataset)));

% read in the spam list
spam_list_path = '/home/phuc/Research/danilo/mv1m/spam_vids.list';
fid = fopen(spam_list_path);
spam_list = textscan(fid, '%s');
spam_list = spam_list{1};
fclose(fid);
[~, in_names, ~] = intersect(new_imdb.images.name, spam_list);
new_imdb.images.name(in_names) = [];
new_imdb.images.id(in_names) = [];
new_imdb.images.set(in_names) = [];
new_imdb.images.label(:, in_names) = [];
save(new_imdb_path, '-struct', 'new_imdb');
