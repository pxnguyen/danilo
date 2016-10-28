function run_danilito
addpath(genpath('MexConv3D'));
opts.expDir = '/mnt/large/pxnguyen/cnn_exp/danilito/';
if ~exist(fullfile(opts.expDir, 'onetenth_imdb.mat'), 'file')
  make_mini_imdb(opts.expDir);
end
opts.train = struct();
opts.train.gpus = [1];
opts.imdbPath = fullfile(opts.expDir, 'onetenth_imdb.mat');

cnn_mv1m(opts)

function make_mini_imdb(exp_dir)
  % construct the mini dataset
danilo_imdb_path = fullfile(exp_dir, 'danilo_imdb.mat');
danilito_imdb_path = fullfile(exp_dir, 'onetenth_imdb.mat');
danilo_imdb = load(danilo_imdb_path);
danilito_imdb = danilo_imdb;
video_indeces = cell(length(danilo_imdb.classes.name), 1);
for class_index = 1:length(danilo_imdb.classes.name)
  class_indeces = danilo_imdb.images.label(class_index, :);
  train_indeces = (danilo_imdb.images.set == 1);
  
  % getting the train indeces
  train_vid_indeces = find(class_indeces & train_indeces);
  random_order = randperm(numel(train_vid_indeces));
  num_take = ceil(numel(train_vid_indeces)/10);
  train_vid_indeces = train_vid_indeces(random_order(1:num_take));
  test_vid_indeces = find(class_indeces & ~train_indeces); % test indeces
  num_train = numel(train_vid_indeces);
  num_test = numel(test_vid_indeces);
  video_indeces{class_index} = cat(2, [train_vid_indeces test_vid_indeces]);
  class_name = danilo_imdb.classes.name{class_index};
  fprintf('%s %d %d\n', class_name, num_train, num_test);
end

video_indeces = unique([video_indeces{:}]);
danilito_imdb.images.name = danilo_imdb.images.name(video_indeces);
danilito_imdb.images.label = danilo_imdb.images.label(:, video_indeces);
danilito_imdb.images.set = danilo_imdb.images.set(video_indeces);
save(danilito_imdb_path, '-struct', 'danilito_imdb');
