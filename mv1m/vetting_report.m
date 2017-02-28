info = load('/mnt/large/pxnguyen/cnn_exp/aria/info.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');
selected_AP = info.AP_tag(aria_imdb.selected);
selected_tags = aria_imdb.classes.name(aria_imdb.selected);
[~, sorted_order] = sort(selected_AP, 'descend');
x = load('/home/phuc/Research/yaromil/vetted_labels.mat');
vetted_labels = x.vetted_labels;

%% how many videos are vetted
vetted = (vetted_labels==2) | (vetted_labels==-2) | (vetted_labels==0);
vetted_positives = (vetted_labels==2);
sum(vetted_positives(:))
vetted_negatives = (vetted_labels==-2);
sum(vetted_negatives(:))
tags_have_vetted = sum(vetted, 2) > 0;

%% how many videos are converted
% from P to P* and N*
test = (aria_imdb.images.set==2);
test_labels = full(aria_imdb.images.label(:, test));
positives = double(vetted_labels) .* test_labels;
p2vp = (positives==2);
p2vn = (positives==-2);

negatives = double(vetted_labels) .* double(test_labels==0);
n2vp = (negatives==2);
n2vn = (negatives==-2);

%% output to prec@k to a file for displaying
resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_self_learn/resdb.mat');

%%
tags_vetted = aria_imdb.classes.name(tags_have_vetted);
scores = resdb.fc1000.outputs;
scores_vetted = scores(tags_have_vetted, :);
original_gts = test_labels(tags_have_vetted, :);
vetted_gts = vetted_labels(tags_have_vetted, :) > 0;
original_prec = compute_precision_at_k(scores_vetted, original_gts, 'k', 16);
adjusted_prec = compute_precision_at_k(scores_vetted, vetted_gts, 'k', 16);

%% print to csv
train_vid_count = full(info.train_vid_count(tags_have_vetted));
fid = fopen('prec_at_16.original.adjusted.csv', 'w');
fprintf(fid, 'tag,num_train,AP,dataset\n');
for index = 1:length(tags_vetted)
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags_vetted{index}, train_vid_count(index),...
    original_prec(index), 'original');
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags_vetted{index}, train_vid_count(index),...
    adjusted_prec(index), 'adjusted');
end
fclose(fid);

%% analyses for training set
train = (aria_imdb.images.set==1);
train_images = aria_imdb.images.name(train);
train_vid_count = sum(aria_imdb.images.label(:, train), 2);
x = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels = x.vetted_labels;
vetted = (vetted_labels==2) | (vetted_labels==-2) | (vetted_labels==0);
tags_have_vetted = sum(vetted, 2) > 0;

%%
train = (aria_imdb.images.set==1);
train_labels = aria_imdb.images.label(:, train);
prob = resdb.fc1000.outputs;
selected_indeces = find(tags_have_vetted);
num_vid_train = train_vid_count(selected_indeces);
training_precs = zeros(numel(selected_indeces), 1);
ventured_precs = zeros(numel(selected_indeces), 1);
for index = 1:length(selected_indeces)
  selected_index = selected_indeces(index);
  tag_name = aria_imdb.classes.name{selected_index};
  gts = logical(full(train_labels(selected_index, :)));
  tag_prob = prob(selected_index, :); % videos do not have this tag
  training_prec = compute_precision_at_k(tag_prob, gts, 'k', 32);
  training_precs(index) = training_prec;
  
  nolabel_images = train_images(~gts);
  vetted_gts = vetted_labels(selected_index, ~gts);
  
  fprintf('working on %s\n', tag_name);
  tag_prob = prob(selected_index, ~gts); % videos do not have this tag
  ventured_adjusted_prec_at_32(index) = compute_precision_at_k(tag_prob, vetted_gts==2, 'k', 32);
  ventured_adjusted_prec_at_16(index) = compute_precision_at_k(tag_prob, vetted_gts==2, 'k', 16);
  ventured_adjusted_prec_at_8(index) = compute_precision_at_k(tag_prob, vetted_gts==2, 'k', 8);
  ventured_adjusted_prec_at_4(index) = compute_precision_at_k(tag_prob, vetted_gts==2, 'k', 4);
  ventured_adjusted_prec_at_2(index) = compute_precision_at_k(tag_prob, vetted_gts==2, 'k', 2);
end

%%
precs = [ventured_adjusted_prec_at_32, ventured_adjusted_prec_at_16, ventured_adjusted_prec_at_8];

pos = ventured_adjusted_prec_at_8 - ventured_adjusted_prec_at_16;
neg = ventured_adjusted_prec_at_16 - ventured_adjusted_prec_at_32;

errorbar(num_vid_train, ventured_adjusted_prec_at_16,...
  pos, neg, 'x');

%%
semilogx(num_vid_train, ventured_adjusted_prec_at_32, 'rx', 'MarkerSize', 10);
hold on
semilogx(num_vid_train, ventured_adjusted_prec_at_16, 'gx', 'MarkerSize', 10);
semilogx(num_vid_train, ventured_adjusted_prec_at_8, 'bx', 'MarkerSize', 10);
grid on;
xlabel('num train')
ylabel('ventured prec@32')

%%
plot(training_precs, ventured_precs, 'rx', 'MarkerSize', 10)
grid on;
xlabel('training prec@32')
ylabel('ventured prec@32')

%%
train_vid_count = full(info.train_vid_count(tags_have_vetted));
fid = fopen('prec_at_16.original.adjusted.csv', 'w');
fprintf(fid, 'tag,num_train,AP,dataset\n');
for index = 1:length(tags_vetted)
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags_vetted{index}, train_vid_count(index),...
    original_prec(index), 'original');
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags_vetted{index}, train_vid_count(index),...
    adjusted_prec(index), 'adjusted');
end
fclose(fid);