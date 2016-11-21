function query_both(resdb, varargin)
run('~/Research/vlfeat-0.9.20/toolbox/vl_setup.m');
opts = struct();
opts.start_exp = 'ari_nospam_small';
opts.database = 'danilo_nospam';
opts = vl_argparse(opts, varargin) ;
root_exp_dir = '/mnt/large/pxnguyen/cnn_exp/';
start_dir = fullfile(root_exp_dir, opts.start_exp);
database_dir = fullfile(root_exp_dir, opts.database);

fprintf('Loading the imdbs...\n');
start_imdb = load(fullfile(start_dir, sprintf('%s_imdb.mat', opts.start_exp)));
database_imdb = load(fullfile(database_dir, sprintf('%s_imdb.mat', opts.database)));
db2start_map = make_map(start_imdb, database_imdb);
fprintf('Done\n')

% compute the difference in training examples
% this is going to be amount to add
fprintf('Computing the amount to add\n');
%diff_counts = compute_train_diff(start_imdb, database_imdb);
fprintf('Done\n')

database_labels = database_imdb.images.label(:, resdb.video_ids);

kmeans = load(sprintf('kmeanres_%s.mat', opts.start_exp));
database_pool5 = resdb.pool5.outputs;
prob = resdb.fc1000.outputs; % the probability belonging to a tag
database_names = resdb.names(resdb.video_ids);
database_tags = resdb.tags;
added_names = cell(numel(start_imdb.classes.name), 1);
added_labels = cell(numel(start_imdb.classes.name), 1);

start_train = (start_imdb.images.set==1);
train_counts = sum(start_imdb.images.label(:, start_train), 2);

for tag_index = 1:numel(start_imdb.classes.name)
  tag_name = start_imdb.classes.name{tag_index};
  fprintf('%d. %s\n', tag_index, tag_name);
  sp = sprintf('selected_both/selected_%s.mat', tag_name(2:end));
  
  if train_counts(tag_index) == 0
    continue
  end
  
  if ~exist(sp, 'file')
    resdb_tag_index = strcmp(tag_name, database_tags); % indexing into the db

    % candidates - keeping track of the videos that are considered
    candidates = true(numel(resdb.video_ids), 1);

    % filter out videos already in start
    start_labels = boolean(start_imdb.images.label(tag_index, :));
    start_names = start_imdb.images.name(start_labels);
    [~, ~, name_overlap_index] = intersect(start_names, resdb.names);
    candidates(name_overlap_index) = false; % remove overlap

    % remove videos that have the same label
    resdb_tag_index = strcmp(tag_name, database_tags);
    same_label = boolean(database_labels(resdb_tag_index, :));
    candidates(same_label) = false; % remove same label

    filtered_names = database_names(candidates);
    filtered_labels = database_labels(:, candidates);

    visual_scores = prob(tag_index, candidates);
    textual_scores = get_text_scores(tag_name, filtered_labels, resdb_tag_index);
    filtered_scores = combine_scores(visual_scores, textual_scores);

    [~, sorted_indeces] = sort(filtered_scores, 'descend');
    selected = sorted_indeces(1:10);

    selected_struct = struct();
    selected_struct.selected_names = filtered_names(selected);

    db_selected_label = logical(database_labels(:, selected));
    ari_selected_label = zeros(807, 10);
    for selected_index = 1:10
      ari_indeces = db2start_map(db_selected_label(:, selected_index));
      ari_indeces(ari_indeces==0) = [];
      ari_selected_label(ari_indeces, selected_index) = 1;
      ari_selected_label(tag_index, selected_index) = 1;
    end
    selected_struct.selected_label = ari_selected_label;
  
    save(sp, '-struct', 'selected_struct');
  else
    selected_struct = load(sp);
  end

  added_names{tag_index} = selected_struct.selected_names;
  added_labels{tag_index} = selected_struct.selected_label;
end

mod_imdb = start_imdb;
mod_imdb.images.id = horzcat(mod_imdb.images.id,...
  (1:numel(added_names)) + 1e7 - 1) ;
mod_imdb.images.name = horzcat(mod_imdb.images.name, added_names) ;
mod_imdb.images.set = horzcat(mod_imdb.images.set, 1*ones(1,numel(added_names))) ;
mod_imdb.images.label = horzcat(mod_imdb.images.label, labels') ;

mod_name = sprintf('ari_mod_both', opts.start_exp);
mod_dir = fullfile(root_exp_dir, mod_name);
mod_imdb_path = fullfile(mod_dir, sprintf('%s_imdb.mat', mod_name));
save(mod_imdb_path, '-struct', 'mod_imdb');

% -------------------------------------------------------------------------
function scores = combine_scores(visual_scores, textual_scores)
% -------------------------------------------------------------------------
%normalizing
visual_scores = visual_scores - mean(visual_scores);
visual_scores = visual_scores / std(visual_scores);
textual_scores = textual_scores - mean(textual_scores);
textual_scores = textual_scores / std(textual_scores);

scores = 0.5 * visual_scores + 0.5 * textual_scores;

% -------------------------------------------------------------------------
function scores=get_text_scores(tag, features, resdb_tag_index)
% -------------------------------------------------------------------------
language_model_dir = '/mnt/large/pxnguyen/cnn_exp/ari_nospam_small/language_models/';
svm_model = load(fullfile(language_model_dir, sprintf('%s.mat', tag(2:end))));
X_copy = features;
X_copy(resdb_tag_index, :) = 0;
scores = svm_model.w' * X_copy + svm_model.b;

% -------------------------------------------------------------------------
function diff_counts = compute_train_diff(start_imdb, database_imdb)
% -------------------------------------------------------------------------
%TODO(phuc): need to find the tags for database_imdb
start_train = (start_imdb.images.set==1);
train_counts = sum(start_imdb.images.label(:, start_train), 2);

database_train_indeces = (database_imdb.images.set==1);
database_tag_indeces = zeros(numel(start_imdb.classes.name), 1);
for tag_index = 1:length(start_imdb.classes.name)
  tag_name = start_imdb.classes.name(tag_index);
  database_tag_indeces(tag_index)= find(strcmp(tag_name, database_imdb.classes.name));
end

database_counts = sum(database_imdb.images.label(...
  database_tag_indeces,...
  database_train_indeces), 2);

diff_counts = database_counts - train_counts;

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

% -------------------------------------------------------------------------
function map_danilo_ari = make_map(ari_imdb, danilo_imdb)
% -------------------------------------------------------------------------
map_danilo_ari = zeros(numel(danilo_imdb.classes.name), 1);
for ari_tag_index = 1:length(ari_imdb.classes.name)
  ari_tag = ari_imdb.classes.name{ari_tag_index};
  danilo_tag_set = danilo_imdb.classes.name;
  danilo_tag_index = strcmp(ari_tag, danilo_tag_set);
  map_danilo_ari(danilo_tag_index) = ari_tag_index;
end
