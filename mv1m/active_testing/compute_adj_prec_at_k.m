function precisions=compute_adj_prec_at_k(imdb, resdb, vetted_labels, varargin)
opts.k_evaluation = 48;
opts.mode = 'per-video';
opts = vl_argparse(opts, varargin);

switch opts.mode
  case 'per-video'
    precisions = compute_per_video(imdb, resdb, vetted_labels, opts);
  case 'per-tag'
    precisions = compute_per_tag(imdb, resdb, vetted_labels, opts);
  otherwise
    error('Unregconized mode: %s\n', mode)
end

function precisions = compute_per_video(imdb, resdb, vetted_labels, opts)
lstruct = load('active_testing/videos.list.mat');
video_ids = lstruct.video_ids;
[video_ids, ~, indeces_to_all] = intersect(video_ids, resdb.video_ids);

% vetted labels, observed labels, classifier scores 1Kx160K
vetted_labels = vetted_labels(imdb.tags_to_train, resdb.video_ids);
observed_label = imdb.images.label(imdb.tags_to_train, resdb.video_ids);
prob = resdb.fc1000.outputs(imdb.tags_to_train, :);

% find the examples and computing the precision
num_vid = numel(video_ids);
precisions = zeros(num_vid, 1);
for index = 1:num_vid
  vid_index = indeces_to_all(index);
  preds = prob(:, vid_index);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_lbls = order(1:opts.k_evaluation);

  % compute the vetted precision
  vetted_lbl_ids = find(abs(vetted_labels(:, vid_index))>1);
  vetted_lbl_ids = intersect(vetted_lbl_ids, topK_lbls);
  vetted_count = full(sum(vetted_labels(vetted_lbl_ids, vid_index)>1));

  % compute the unvetted precision
  unvetted_lbl_id = setdiff(topK_lbls, vetted_lbl_ids);
  pos_prob = observed_label(unvetted_lbl_id, vid_index);
  unvetted_count = full(sum(pos_prob));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
  fprintf('%s: %0.2f %d/%d vetted %d/%d unvetted \n',...
    imdb.images.name{resdb.video_ids(vid_index)}, precisions(index),...
    vetted_count, numel(vetted_lbl_ids), unvetted_count, numel(unvetted_lbl_id));
end

function precisions=compute_per_tag(imdb, resdb, vetted_labels, opts)
fid = fopen('active_testing/nuswide/Concepts81.txt');
tag_set_1000 = imdb.classes.name;
tags = textscan(fid, '%s\r\n');% tags = tags{1};
tag_indeces_b1000_bin = false(1000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(tag_set_1000, tags{index});
  tag_indeces_b1000_bin(tag_index) = true;
end
tag_indeces_1000 = find(tag_indeces_b1000_bin);

% vetted labels
vetted_labels = vetted_labels(:, resdb.video_ids);
observed_label = imdb.images.label(:, resdb.video_ids);

prob = resdb.fc1000.outputs(:, :);

% find the examples and computing the precision
num_tag = numel(tag_indeces_1000);
precisions = zeros(num_tag, 1);
for index = 1:num_tag
  tag_index_1000 = tag_indeces_1000(index);
  tag_name = tag_set_1000{tag_index_1000};
  preds = prob(tag_index_1000, :);
  [~, order] = sort(preds, 'descend');

  % get the top k videos
  topK_videoid = order(1:opts.k_evaluation);

  % compute the vetted precision
%   vetted_videoid = vetted_labels(tag_index_1000, :);
%   vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(full(vetted_labels(tag_index_1000, topK_videoid)));

  % compute the unvetted precision
%   unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
%   pos_prob = observed_label(tag_index_1000, unvetted_videoid);
%   unvetted_count = full(sum(pos_prob));
  
  precisions(index) = (vetted_count)/opts.k_evaluation;
%   fprintf('%s: %d/%d vetted %d/%d unvetted, prec@%d: %0.4f\n', tag_name,...
%     vetted_count, numel(vetted_videoid), unvetted_count, numel(unvetted_videoid), opts.k_evaluation,...
%     precisions(index));
end
