function write_out_prediction(exp_name, resdb)
% write out the top k predictions for an experiment
% Args
%   exp_name: the experiment name that was runned
opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
opts.imdbPath = fullfile(opts.expDir, sprintf('%s_imdb.mat', exp_name));
opts.k = 64;
[epoch, iter] = findLastCheckpoint(opts.expDir);
opts.resdb_path = fullfile(opts.expDir,...
  sprintf('resdb-iter-%d.mat', iter));
opts.model_path = fullfile(opts.expDir,...
  sprintf('net-epoch-%d-iter-%d.mat', epoch, iter));
imdb = load(opts.imdbPath);

fid = fopen('active_testing/tags.list');
tags = textscan(fid, '%s\n');
tags = tags{1};
tag_indeces_bin = false(4000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(imdb.classes.name, tags{index});
  tag_indeces_bin(tag_index) = true;
end

tag_indeces = find(tag_indeces_bin);

fprintf('Loading imdb\n');
tic; imdb = load(opts.imdbPath); toc;
print_out_perimage(resdb, imdb, tag_indeces, exp_name)

% -------------------------------------------------------------------------
function print_out_perimage(resdb, imdb, tag_indeces, exp_name)
% -------------------------------------------------------------------------
test_images = resdb.video_ids;
num_videos = numel(test_images);
prob = vl_nnsigmoid(resdb.fc1000.outputs);
for index = 1:num_videos
  video_id = test_images(index);
  [tag_scores, tag_order] = sort(prob(:, index), 'descend');
  tag_order = tag_order(1:32);
  tag_scores = tag_scores(1:32);
%   [tag_order, ia, ~] = intersect(tag_order, tag_indeces);
%   tag_scores = tag_scores(ia);
  tag_names = imdb.classes.name(tag_order);

  video_name = imdb.images.name{video_id};
  fprintf('working on %s (%d/%d)\n', video_name, index, num_videos);
  [~, video_name, ~] = fileparts(video_name);
  folder_name = sprintf('prediction_pervideo_dir/%s', exp_name);
  if ~exist(folder_name, 'dir'); mkdir(folder_name); end;
  file_name = fullfile(folder_name, sprintf('%s.txt', video_name));
  if exist(file_name, 'file'); continue; end;
  fid = fopen(file_name, 'w');
  for i=1:32
    fprintf(fid, '%s,%0.4f\n', tag_names{i}(2:end), tag_scores(i));
  end
  fclose(fid);
end

% -------------------------------------------------------------------------
function [epoch, iter] = print_out_pertag(modelDir)
% -------------------------------------------------------------------------
test_images = resdb.video_ids;
prob = resdb.fc1000.outputs;
selected_indeces = find(imdb.selected);
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = imdb.classes.name{selected_index};
  
  fprintf('working on %s\n', tag_name);
  tag_prob = prob(selected_index, :); % videos do not have this tag
  [~, order] = sort(tag_prob, 'descend');
  dir_name = fullfile('prediction_videos_dir', exp_name);
  if ~exist(dir_name, 'dir')
    mkdir(dir_name);
  end
  
  file_path = fullfile(dir_name,...
    sprintf('%s.test', tag_name(2:end)));
  fid = fopen(file_path, 'w');
  image_list = imdb.images.name(test_images(order(1:opts.k)));
  
  for image_index = 1:length(image_list)
    [~, file, ~] = fileparts(image_list{image_index});
    fprintf(fid, '%s\n', file);
  end
  fclose(fid);
end

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
