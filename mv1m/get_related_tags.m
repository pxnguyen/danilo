function related_video_collection=get_related_tags(imdb)
% given an imdb, find all the related tags for each tags
train_indeces = (imdb.images.set==1);
tag_set = imdb.classes.name;
related_video_collection = cell(numel(tag_set), 1);
related_file_dir = '/home/phuc/Research/yaromil/yaromil/static/filelist';
for tag_index=1:numel(tag_set)
  tag_name = imdb.classes.name{tag_index};
  write_file_path = fullfile(related_file_dir, sprintf('%s_related_vids.list', tag_name(2:end)));
  fid = fopen(write_file_path, 'w');
  video_with_tag_indeces = imdb.images.label(tag_index, :) & train_indeces;
  related_tags_count = sum(imdb.images.label(:, video_with_tag_indeces), 2);
  [num_vid, sorted_indeces] = sort(related_tags_count, 'descend');
  max_related = max(find(num_vid > 0));
  related_tag_indeces = sorted_indeces(1:min(10, max_related));
  related_video_bin = sum(imdb.images.label(related_tag_indeces, :)) > 0;
  related_videos = find(related_video_bin & train_indeces & ~video_with_tag_indeces);
  related_video_collection{tag_index} = related_videos;
  post_ids = cellfun(@(name) get_post_id(name), imdb.images.name(related_videos),...
    'UniformOutput', false);
  for post_id=post_ids
    fprintf(fid, '%s\n', post_id{1});
  end
  fclose(fid);
  fprintf('%d. %s\n', tag_index, tag_name);
  %fprintf('%s: stock %d related %d\n', tag_name,...
    %sum(full(video_with_tag_indeces)), numel(related_videos));
end

function postid=get_post_id(name)
[~, postid, ~] = fileparts(name);
