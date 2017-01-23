info = load('/mnt/large/pxnguyen/cnn_exp/aria/info.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');
selected_AP = info.AP_tag(aria_imdb.selected);
selected_tags = aria_imdb.classes.name(aria_imdb.selected);
[~, sorted_order] = sort(selected_AP, 'descend');
train = (aria_imdb.images.set==1);
test = (aria_imdb.images.set==2);

%%
fid = fopen('tag_list', 'w');
for tag_index = 1:numel(aria_imdb.classes.name)
  tag = aria_imdb.classes.name{tag_index}(2:end);
  fprintf(fid, '%s\n', tag);
end
fclose(fid);

%% print out the testing list
test = (aria_imdb.images.set==2);
test_images = aria_imdb.images.name(test);
fid2 = fopen('video.test.list', 'w');
for image_index = 1:numel(test_images)
  [~, file, ~] = fileparts(test_images{image_index});
  fprintf(fid2, '%s\n', file);
end
fclose(fid);

%% print out the training list
train = (aria_imdb.images.set==1);
train_images = aria_imdb.images.name(train);
fid2 = fopen('video.train.list', 'w');
for image_index = 1:numel(train_images)
  [~, file, ~] = fileparts(train_images{image_index});
  fprintf(fid2, '%s\n', file);
end
fclose(fid);

%% print out the tags
fid = fopen('tag_list.order_by_ap', 'w');
for index=1:numel(sorted_order)
  sorted_index = sorted_order(index);
  fprintf(fid, '%s,%0.3f\n', selected_tags{sorted_index}, selected_AP(sorted_index));
end

%% loading resdb for aria visual
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-646000.mat');

%% loading resdb for aria visual
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_desc/resdb-iter-404000.mat');
storage_path = 'aria_desc';

%%
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_self_learn/resdb.mat');
storage_path = 'aria_train';

%%
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_self_learn/resdb.mat');
storage_path = 'aria_train_all';

%% make the imdb with the top-8
aria_top8_imdb_path = '/mnt/large/pxnguyen/cnn_exp/aria_top8/aria_top8_imdb.mat';
aria_top8_imdb = aria_imdb;
test_images = resdb.video_ids;
aria_top8_imdb.tags_to_train = aria_imdb.selected;
prob = resdb.fc1000.outputs;
selected_indeces = find(aria_imdb.selected);
k = 8;
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = aria_imdb.classes.name{selected_index};
  gts = logical(resdb.gts(selected_index, :));
  test_images_tag = test_images(~gts);
  
  fprintf('working on %s\n', tag_name);
  tag_prob = prob(selected_index, ~gts); % videos do not have this tag
  [scores_order, order] = sort(tag_prob, 'descend');
  toflip = test_images_tag(order(1:k));
  aria_top8_imdb.images.label(selected_index, toflip) = 1;  
end

save(aria_top8_imdb_path, '-struct', 'aria_top8_imdb')

%%
test_images = resdb.video_ids;
prob = resdb.fc1000.outputs;
selected_indeces = find(aria_imdb.selected);
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = aria_imdb.classes.name{selected_index};
  
  fprintf('working on %s\n', tag_name);
  tag_prob = prob(selected_index, :); % videos do not have this tag
  [~, order] = sort(tag_prob, 'descend');
  file_path = fullfile('prediction_videos_dir', storage_path,...
    sprintf('%s.test', tag_name(2:end)));
  fid = fopen(file_path, 'w');
  image_list = aria_imdb.images.name(test_images(order(1:32)));
  
  for image_index = 1:length(image_list)
    [~, file, ~] = fileparts(image_list{image_index});
    fprintf(fid, '%s\n', file);
  end
end

%%
test_images = resdb.video_ids;
prob = resdb.fc1000.outputs;
selected_indeces = find(aria_imdb.selected);
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = aria_imdb.classes.name{selected_index};
  gts = logical(resdb.gts(selected_index, :));
  test_images_tag = test_images(~gts);
  
  fprintf('working on %s\n', tag_name);
  tag_prob = prob(selected_index, ~gts); % videos do not have this tag
  [~, order] = sort(tag_prob, 'descend');
  file_path = fullfile('prediction_videos_dir', storage_path,...
    sprintf('%s.test', tag_name(2:end)));
  fid = fopen(file_path, 'w');
  image_list = aria_imdb.images.name(test_images_tag(order(1:32)));
  
  for image_index = 1:length(image_list)
    [~, file, ~] = fileparts(image_list{image_index});
    fprintf(fid, '%s\n', file);
  end
end


%% print out list
% most popular
% least popular
% median popularity
train = (aria_imdb.images.set==1);
test = (aria_imdb.images.set==2);
train_labels = aria_imdb.images.label(aria_imdb.selected, train);
counts = sum(train_labels, 2);
[~, count_order] = sort(counts, 'descend');
fprintf(strjoin(selected_tags(count_order(1:10)), ' '))
fprintf('\n');
fprintf(strjoin(selected_tags(count_order(end-10:end)), ' '))
fprintf('\n');
fprintf(strjoin(selected_tags(count_order(495:505)), ' '))

% highest APs
% lowest APs
% median APs
[~, ap_order] = sort(selected_AP, 'descend');
fprintf(strjoin(selected_tags(ap_order(1:10)), ' '))
fprintf('\n');
fprintf(strjoin(selected_tags(ap_order(end-10:end)), ' '))
fprintf('\n');
fprintf(strjoin(selected_tags(ap_order(400:600)), ' '))

%% test images
test_images = aria_imdb.images.name(test);
test_labels = aria_imdb.images.label(:, test);
selected_indeces = find(aria_imdb.selected);
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = aria_imdb.classes.name{selected_index};
  fprintf('working on %s\n', tag_name);
  file_path = fullfile('test_videos_dir',...
    sprintf('%s.test', tag_name(2:end)));
  fid = fopen(file_path, 'w');
  images_tag_bin = boolean(test_labels(selected_index, :));
  image_list = test_images(images_tag_bin);
  
  for image_index = 1:length(image_list)
    [~, file, ~] = fileparts(image_list{image_index});
    fprintf(fid, '%s\n', file);
  end
end

%%
for i=1:10
  ap = selected_AP(sorted_order(i));
  ap = uint32(ap*100);
  tag = selected_tags{sorted_order(i)};
  tags = cell(ap, 1);
  [tags{1:ap}] = deal(tag);
  fprintf('%s\n', strjoin(tags, ' '));
end

%%
for i=1:100
  ap = selected_AP(sorted_order(end-i+1));
  ap = uint32(ap*100000);
  tag = selected_tags{sorted_order(end-i+1)};
  tags = cell(ap, 1);
  [tags{1:ap}] = deal(tag);
  fprintf('%s\n', strjoin(tags, ' '));
end
%% median
for i=1:100
  ap = selected_AP(sorted_order(450+i));
  ap = uint32(ap*1000);
  tag = selected_tags{sorted_order(450+i)};
  tags = cell(ap, 1);
  [tags{1:ap}] = deal(tag);
  fprintf('%s\n', strjoin(tags, ' '));
end
%%