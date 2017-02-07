vis_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_copy/resdb-iter-1.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');


%%
tags = {'#cat', '#dogs', '#bird', '#crafts', '#nba', '#trump'};
tags_indeces = zeros(numel(tags), 1);
for index = 1:numel(tags)
  tags_indeces(index) = find(strcmp(aria_imdb.classes.name, tags{index}));
end

%%
labels = aria_imdb.images.label(tags_indeces, vis_resdb.video_ids);
relevant_vids = sum(labels, 1)>0;
relevant_video_ids = vis_resdb.video_ids(relevant_vids);

%%
video_ids = vis_resdb.video_ids(relevant_vids);
feature = vis_resdb.pool5.outputs(:, relevant_vids);
[idx] = knnsearch(feature', feature(:, 1:600)', 'K', 10);