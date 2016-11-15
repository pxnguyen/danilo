function get_related_tags(imdb)
% given an imdb, find all the related tags for each tags
train_indeces = (imdb.images.set==1);
for tag_index=1:numel(imdb.classes.name)
  tag_name = imdb.classes.name{tag_index};
  video_with_tag_indeces = imdb.images.label(tag_index, :) & train_indeces;
  related_tags_count = sum(imdb.images.label(:, video_with_tag_indeces), 2);
  [num_vid, sorted_indeces] = sort(related_tags_count, 'descend');
  related_tags = strjoin(imdb.classes.name(sorted_indeces(1:10)), ' ');
  fprintf('%s: %s\n', tag_name, related_tags);
end
