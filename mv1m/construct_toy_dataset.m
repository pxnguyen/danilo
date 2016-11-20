function toy_imdb=construct_toy_dataset
%load imdb
existing_imdb_path = '/mnt/large/pxnguyen/cnn_exp/danilo/imdb.mat';
toy_imdb_path = '/mnt/large/pxnguyen/cnn_exp/danilo/toy_imdb.mat';
imdb = load(existing_imdb_path);

% get the tags indeces
toy_tags = {'#cat', '#dog', '#dancing', '#justinbieber', '#suicidesquad', '#pokemon', '#disney'};
toy_tags_indeces = zeros(length(toy_tags), 1);
for tag_index = 1:length(toy_tags)
  tag = toy_tags{tag_index};
  toy_tags_indeces(tag_index) = find(strcmp(tag, imdb.classes.name));
end

toy_imdb = struct();

% only select the videos with this indeces
num_videos = 700;
names = cell(1, length(toy_tags));
labels = zeros(length(toy_tags), num_videos*length(toy_tags));
for tag_index = 1:length(toy_tags)
  video_indeces = find(imdb.images.label(toy_tags_indeces(tag_index), :));
  random_order = randperm(length(video_indeces));
  video_indeces = video_indeces(random_order(1:num_videos));
  names{tag_index} = video_indeces;
end

video_indeces = cat(2, names{:});
video_indeces = unique(video_indeces);
names = imdb.images.name(video_indeces);
labels = imdb.images.label(toy_tags_indeces, video_indeces);

toy_imdb.images.id = 1:numel(names) ;
toy_imdb.images.name = names ;
toy_imdb.images.set = randi(2, 1, numel(names));
%toy_imdb.images.set = ones(1, numel(names));
toy_imdb.images.label = sparse(labels) ;

%toy_imdb.images.id = horzcat(toy_imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
%toy_imdb.images.name = horzcat(toy_imdb.images.name, names) ;
%toy_imdb.images.set = horzcat(toy_imdb.images.set, 2*ones(1,numel(names))) ;
%toy_imdb.images.label = sparse(horzcat(labels, labels));

toy_imdb.classes.name = toy_tags;

save(toy_imdb_path, '-struct', 'toy_imdb')

function tag = extract_tag_list(name)
outs = strsplit(name, ',');
tag = outs{2};
