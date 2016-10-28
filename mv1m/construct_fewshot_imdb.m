function construct_fewshot_imdb(ari_type)
danilo_dir = '/mnt/large/pxnguyen/cnn_exp/danilo/';
imdb = load(fullfile(danilo_dir, 'imdb.mat'));
aggregate_results_path = fullfile(danilo_dir, 'aggregate_results.mat');
aggr_results = load(aggregate_results_path);
criteria = aggr_results.danilo_AP > aggr_results.danilito_AP;
criteria = criteria & (aggr_results.danilo_AP > aggr_results.chance_AP);
improvements = (aggr_results.danilo_AP./aggr_results.danilito_AP - 1) * 100;
% make a new imdb
selected_tags = find(improvements > 50 & criteria);
video_indeces = cell(numel(selected_tags), 1);
train_indeces = (imdb.images.set==1);
eval_indeces = find(imdb.images.set==2);
for i = 1:numel(selected_tags)
  tag_index = selected_tags(i);
  class_indeces = imdb.images.label(tag_index, :);
  train_class_indeces = find(class_indeces & train_indeces);
  random_order = randperm(numel(train_class_indeces));
  if strcmp(ari_type, 'full')
    num_take = ceil(numel(random_order));
  elseif strcmp(ari_type, 'small')
    num_take = ceil(numel(random_order)/10);
  end
  fprintf('%d %s %d\n', i, imdb.classes.name{tag_index}, num_take);
  video_indeces{i} = train_class_indeces(random_order(1:num_take));
end

video_indeces = unique(cat(2, video_indeces{:}));
names = imdb.images.name(video_indeces);
labels = imdb.images.label(selected_tags, video_indeces);
new_imdb = struct();
new_imdb.classes.name = imdb.classes.name(selected_tags);
new_imdb.imageDir = imdb.imageDir;

new_imdb.images.id = 1:numel(names) ;
new_imdb.images.name = names ;
new_imdb.images.set = ones(1, numel(names)) ; % train
new_imdb.images.label = labels ;

eval_names = imdb.images.name(eval_indeces);
eval_labels = imdb.images.label(selected_tags, eval_indeces);
new_imdb.images.id = horzcat(new_imdb.images.id, (1:numel(eval_names)) + 1e7 - 1) ;
new_imdb.images.name = horzcat(new_imdb.images.name, eval_names) ;
new_imdb.images.set = horzcat(new_imdb.images.set, 2*ones(1,numel(eval_names))) ;
new_imdb.images.label = horzcat(new_imdb.images.label, eval_labels) ;

ari_dir = '/mnt/large/pxnguyen/cnn_exp/ari/';
new_imdb_path = fullfile(ari_dir, sprintf('ari_%s_imdb.mat', ari_type));
save(new_imdb_path, '-struct', 'new_imdb');
