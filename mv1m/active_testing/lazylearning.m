%% load the resdb with pool 5
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-1632000-test.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');

pool5 = resdb.pool5.outputs;

%%
cnn_exp = '/mnt/large/pxnguyen/cnn_exp/';
resdb_A = load(fullfile(cnn_exp, 'aria/resdb-iter-390000-test.mat'));
prob = resdb_A.fc1000.outputs(aria_imdb.tags_to_train, :);

%%
vetted_labels_train = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load('/home/phuc/Research/yaromil/vetted_labels_test.mat');
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

%% get the top video list
fid = fopen('active_testing/tags.list');
tag_set_1000 = aria_imdb.classes.name(aria_imdb.selected);
tags = textscan(fid, '%s\n'); tags = tags{1};
tag_indeces_bin = false(4000, 1);
tag_indeces_b1000_bin = false(1000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(aria_imdb.classes.name, tags{index});
  tag_indeces_bin(tag_index) = true;
  
  tag_index = strcmp(tag_set_1000, tags{index});
  tag_indeces_b1000_bin(tag_index) = true;
end
tag_indeces_1000 = find(tag_indeces_b1000_bin);

%%
label_set.vetted_labels = vetted_labels(aria_imdb.tags_to_train, resdb.video_ids);
label_set.observed_label = aria_imdb.images.label(aria_imdb.tags_to_train, resdb.video_ids);
num_tag = numel(tag_indeces_1000);
top_videos_all = cell(num_tag, 1);
top_videos_resdb_indeces = cell(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);

  cat_prob = prob(tag_index_1000, :);
  [shortlist_scores, shortlist_order] = sort(cat_prob, 'descend');
  topk_videoids_imdb = shortlist_order(1:48);
  top_videos_all{index} = topk_videoids_imdb;
end
top_videos_all = cat(2, top_videos_all{:});
top_videos_all = unique(top_videos_all);

%% get pool 5 for the top videos
pool5_top = struct();
pool5_top.video_ids = top_videos_all;
pool5_top.pool5 = resdb.pool5.outputs(:, pool5_top.video_ids);
save('active_testing/pool5_top.mat', '-struct', 'pool5_top');

%% compute the image distance
pool5_kernel = vl_alldist(pool5_top.pool5, pool5_top.pool5, 'L2');
sigma = 1;
sample_distance = exp(-pool5_kernel/(2*sigma^2));
save('active_testing/pool5_kernel.mat', 'pool5_kernel');

%% compute the label distance
topk_obs_lbls = label_set.observed_label(:, top_videos_all);
l2 = vl_alldist2(topk_obs_lbls, topk_obs_lbls, 'L2');
sigma = 1;
label_distance = exp(-l2/(2*sigma^2));
save('active_testing/obs_lbls_kernel.mat', 'label_distance');

%% total_distance
total_similarity = struct();
total_similarity.value = pool5_kernel.* label_distance;
total_similarity.video_ids = pool5_top.video_ids;
save('active_testing/total_similarity.mat', '-struct', 'total_similarity');

%% compute the kernel matrix for them

% save the kernel matrix

%% add this into the current pipeline