function [indeces, imdb]=select_training_examples(num_train, imdb)
% Select the training examples with class-balancing
% Args:
%   num_train: the number of training example to query
%   imdb: the image database
% Output:
%   indeces: the location of the selection
is_train = (imdb.images.set==1);
labels = imdb.images.label;
num_tag = numel(imdb.classes.name);
multiple = 5; % this helps speed up the selection process
tag_indeces = randi(num_tag, [1 num_train/multiple]);
indeces = zeros(num_train, 1);
augmented_labels = imdb.images.label;
for index = 1:numel(tag_indeces) % for each selected tag
  tag = tag_indeces(index);
  vids_with_tag = find(labels(tag, :) & is_train);
  if isempty(vids_with_tag)
    selected_vids = randi(sum(is_train), [1 multiple]);
  else
    selected_vids = vids_with_tag(randi(numel(vids_with_tag), [1 multiple]));
  end
  indeces(multiple*(index-1)+1:multiple*index) = selected_vids;
  
  % only allow the tag to be true.
  augmented_labels(:, selected_vids) = false;
  augmented_labels(tag, selected_vids) = true;
end

% shuffle the indeces
indeces = indeces(randperm(numel(indeces)));

imdb.images.augmented_labels = augmented_labels;