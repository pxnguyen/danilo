function query_pool5(resdb)
run('~/Research/vlfeat-0.9.20/toolbox/vl_setup.m');
root_exp_dir = '/mnt/large/pxnguyen/cnn_exp/';
ari_small_dir = fullfile(root_exp_dir, 'ari_small');
danilo_dir = fullfile(root_exp_dir, 'danilo');

fprintf('Loading the imdbs...\n');
ari_small_imdb = load(fullfile(ari_small_dir, 'ari_small_imdb.mat'));
danilo_imdb = load(fullfile(danilo_dir, 'danilo_imdb.mat'));
%[epoch, iter] = findLastCheckpoint(danilo_dir);

%fprintf('Loading the database...\n');
%opts.resdb_path = fullfile(danilo_dir,...
%  sprintf('resdb-pool5-model-%s-dataset-%s-iter-%d.mat', 'ari_small', database, iter));
%keyboard
%resdb = load(opts.resdb_path);
resdb.video_ids = cat(2, resdb.video_ids{:});
labels = danilo_imdb.images.label(:, resdb.video_ids);

% load the centroids
kmeans = load('kmeanres_ari_small.mat');
prob = resdb.sigmoid.outputs;
database_names = resdb.names;
database_tags = danilo_imdb.classes.name;
added_names = cell(numel(ari_small_imdb.classes.name), 1);
added_labels = cell(numel(ari_small_imdb.classes.name), 1);
for tag_index = 1:numel(ari_small_imdb.classes.name)
  tag_name = ari_small_imdb.classes.name{tag_index};
  centroid = kmeans.centroids{tag_index};

  ari_labels = boolean(ari_small_imdb.images.label(tag_index, :));
  ari_small_names = ari_small_imdb.images.name(ari_labels);
  [~, ~, database_overlap_index] = intersect(ari_small_names, database_names);

  candidates = true(numel(database_names), 1);

  resdb_tag_index = strcmp(tag_name, database_tags);
  same_label = boolean(labels(resdb_tag_index, :));
  candidates(same_label) = 0; % remove same label
  candidates(database_overlap_index) = 0; % remove overlap
  keyboard

  %distance_to_db = vl_alldist(centroid, pool5_db, 'L2');
  filtered_distance = prob(tag_index, candidates);
  filtered_names = database_names(candidates);
  [~, sorted_indeces] = sort(filtered_distance, 'descend');
  keyboard
  selected = sorted_indeces(1:50);
  filtered_distance(selected)
  added_names{tag_index} = filtered_names(selected);
  added_labels{tag_index} = labels(:, selected);
end

ari_mod_imdb = ari_small_imdb;
ari_mod_imdb.images.id = horzcat(ari_mod_imdb.images.id,...
  (1:numel(added_names)) + 1e7 - 1) ;
ari_mod_imdb.images.name = horzcat(ari_mod_imdb.images.name, added_names) ;
ari_mod_imdb.images.set = horzcat(ari_mod_imdb.images.set, 1*ones(1,numel(added_names))) ;
ari_mod_imdb.images.label = horzcat(ari_mod_imdb.images.label, labels') ;

ari_mod_dir = '/mnt/large/pxnguyen/cnn_exp/ari_mod';
ari_mod_imdb_path = fullfile(root_exp_dir, 'ari_mod', 'ari_mod_imdb.mat');
save(ari_mod_imdb_path, '-struct', 'ari_mod_imdb')

% -------------------------------------------------------------------------
function [epoch, iter] = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*-iter-*.mat')) ;
if ~isempty(list)
  tokens = regexp({list.name}, 'net-epoch-([\d]+)-iter-([\d]+).mat', 'tokens') ;
  epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
  iter = cellfun(@(x) sscanf(x{1}{2}, '%d'), tokens) ;
  epoch = max([epoch 0]);
  iter = max([iter 0]);
else
  epoch = 1;
  iter = 0;
end

