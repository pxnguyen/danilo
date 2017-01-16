function [indeces, imdb, augmented_labels]=select_training_examples(num_train, imdb, varargin)
% Select the training examples with class-balancing
% Args:
%   num_train: the number of training example to query
%   imdb: the image database
%   label_type: original, vetted, ...
% Output:
%   indeces: the location of the selection
opts.label_type = 'original';
opts.loss_type = 'softmax';
opts.is_train = true;
opts = vl_argparse(opts, varargin) ;

is_set = (imdb.images.set==1);

if strcmp(opts.label_type, 'original')
  labels = imdb.images.label;
elseif strcmp(opts.label_type, 'vetted')
  labels = imdb.images.vetted_labels > 0;
end

if isfield(imdb, 'tags_to_train')
  num_tag = sum(imdb.tags_to_train);
  tags_to_train_indeces = find(imdb.tags_to_train);
else
  num_tag = numel(imdb.classes.name);
  tags_to_train_indeces = 1:num_tag;
end

multiple = 4; % this helps speed up the selection process
tag_indeces = randi(num_tag, [1 num_train/multiple]);
tag_indeces = tags_to_train_indeces(tag_indeces);
indeces = zeros(num_train, 1);
if strcmp(opts.loss_type, 'softmax')
  augmented_labels = zeros(num_train, 1);
else
  augmented_labels = labels;
end

for index = 1:numel(tag_indeces) % for each selected tag
  tag = tag_indeces(index);
  vids_with_tag = find(labels(tag, is_set));
  if isempty(vids_with_tag)
    selected_vids = randi(sum(is_set), [1 multiple]);
  else
    selected_vids = vids_with_tag(randi(numel(vids_with_tag), [1 multiple]));
  end
  indeces(multiple*(index-1)+1:multiple*index) = selected_vids;

  % only allow the tag to be true.
  %augmented_labels(:, selected_vids) = false;
  %augmented_labels(tag, selected_vids) = true;
  
  if strcmp(opts.loss_type, 'softmax')
    augmented_labels(multiple*(index-1)+1:multiple*index) = tag;
  end
end

% shuffle the indeces
order = randperm(numel(indeces));
indeces = indeces(order);
augmented_labels = augmented_labels(order);

imdb.images.augmented_labels = augmented_labels;