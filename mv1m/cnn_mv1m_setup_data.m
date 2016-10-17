function imdb = cnn_mv1m_setup_data(varargin)
% Setup the first split 1 for UCF-101
opts.dataDir = '/mnt/large/pxnguyen/vine-large-2/';
opts.lite = false ;
opts = vl_argparse(opts, varargin) ;

% -------------------------------------------------------------------------
%                                                  Load categories metadata
% -------------------------------------------------------------------------

class_ind_file = fullfile(opts.dataDir, 'label_map.csv');
file_id = fopen(class_ind_file);
outputs = textscan(file_id, '%d%s%d');
imdb.classes.name =  outputs{2};
imdb.classes.count =  outputs{3};
imdb.imageDir = fullfile(opts.dataDir, 'videos') ;
fclose(file_id);

% -------------------------------------------------------------------------
%                                                           Training images
% -------------------------------------------------------------------------

fprintf('Getting data for training videos\n');
train_s1_file = fullfile(opts.dataDir, 'train_list.csv');
file_id = fopen(train_s1_file);
outputs = textscan(file_id, '%s%s', 'Delimiter', ',');
names = strcat(outputs{1}, '.mp4')';
labels = outputs{2};
labels = cellfun(@split_labels, labels, 'UniformOutput', false);

imdb.images.id = 1:numel(names) ;
imdb.images.name = names ;
imdb.images.set = ones(1, numel(names)) ;
imdb.images.label = labels' ;

% -------------------------------------------------------------------------
%                                                         Validation images
% -------------------------------------------------------------------------
fprintf('Getting data for testing videos\n');
test_s1_file = fullfile(opts.dataDir, 'test_list.csv');
file_id = fopen(test_s1_file);
outputs = textscan(file_id, '%s%s', 'Delimiter', ',');
names = strcat(outputs{1}, '.mp4')';
labels = outputs{2};
labels = cellfun(@split_labels, labels, 'UniformOutput', false);

imdb.images.id = horzcat(imdb.images.id, (1:numel(names)) + 1e7 - 1) ;
imdb.images.name = horzcat(imdb.images.name, names) ;
imdb.images.set = horzcat(imdb.images.set, 2*ones(1,numel(names))) ;
imdb.images.label = horzcat(imdb.images.label, labels') ;

function labels = split_labels(labels_str)
%Split the label string
labels = strsplit(labels_str, ';');
outlist = zeros(1, length(labels));
for i=1:length(labels)
  outlist(i) = str2num(labels{i});
end
