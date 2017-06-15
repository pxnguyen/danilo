function precisions=compute_true_precision_at_k(imdb, resdb, vetted_labels, varargin)
opts.k_evaluation = 48;
opts = vl_argparse(opts, varargin);

fid = fopen('active_testing/tags.list');
tag_set_1000 = imdb.classes.name(imdb.selected);
tags = textscan(fid, '%s\n'); tags = tags{1};
tag_indeces_b1000_bin = false(1000, 1);
for index = 1:numel(tags)
  tag_index = strcmp(tag_set_1000, tags{index});
  tag_indeces_b1000_bin(tag_index) = true;
end
tag_indeces_1000 = find(tag_indeces_b1000_bin);

% vetted labels
vetted_labels = vetted_labels(imdb.tags_to_train, resdb.video_ids);
observed_label = imdb.images.label(imdb.tags_to_train, resdb.video_ids);

prob = resdb.fc1000.outputs(imdb.tags_to_train, :);

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
  vetted_videoid = find(abs(vetted_labels(tag_index_1000, :))>1);
  vetted_videoid = intersect(vetted_videoid, topK_videoid);
  vetted_count = sum(vetted_labels(tag_index_1000, vetted_videoid)>1);

  % compute the unvetted precision
  unvetted_videoid = setdiff(topK_videoid, vetted_videoid);
  pos_prob = observed_label(tag_index_1000, unvetted_videoid);
  unvetted_count = full(sum(pos_prob));
  
  precisions(index) = (vetted_count + unvetted_count)/opts.k_evaluation;
  fprintf('%s: %d/%d vetted %d/%d unvetted, prec@%d: %0.4f\n', tag_name,...
    vetted_count, numel(vetted_videoid), unvetted_count, numel(unvetted_videoid), opts.k_evaluation,...
    precisions(index));
end