function make_nus_wide_imdb()
fprintf('Loading the labels\n');
labels = load('active_testing/nuswide/nus_wide_labels.mat');
labels = labels.labels;

fprintf('Loading image names\n');
fid = fopen('active_testing/nuswide/image_names.txt');
names = textscan(fid, '%s\n'); names = names{1};

fprintf('tag names\n');
fid = fopen('active_testing/nuswide/Final_Tag_List.txt');
tags = textscan(fid, '%s\r\n'); tags = tags{1};

fprintf('Loading the set\n');
set = load('active_testing/nuswide/set.mat');
imdb = struct();
imdb.classes.name = tags';
imdb.images.set = set.splitset;
imdb.images.name = names';
imdb.images.label = labels;

% vetted_labels_train = importdata('active_testing/nuswide/Train_Tags81.txt');
% vetted_labels_test = importdata('active_testing/nuswide/Test_Tags81.txt');
% vetted_labels = [vetted_labels_train; vetted_labels_test];
% imdb.vetted_labels = imdb.vetted_labels';

fid = fopen('active_testing/nuswide/Concepts81.txt');
vetted_tags = textscan(fid, '%s\r\n'); vetted_tags = vetted_tags{1};
imdb.vetted_tags = vetted_tags';
imdb.vetted_labels = zeros(81, size(labels, 2));

for i_tag = 1:length(imdb.vetted_tags)
  tag = imdb.vetted_tags{i_tag};
  fp = sprintf('active_testing/nuswide/AllLabels/Labels_%s.txt', tag);
  imdb.vetted_labels(i_tag, :) = importdata(fp);
end

pathdir = '/mnt/large/pxnguyen/cnn_exp/nuswide/';
if ~exist(pathdir, 'dir'); mkdir(pathdir); end
imdb_path = fullfile(pathdir, 'nuswide_imdb.mat');
save(imdb_path, '-struct', 'imdb');
fprintf('saved to %s\n', imdb_path);