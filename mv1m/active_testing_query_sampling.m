info = load('/mnt/large/pxnguyen/cnn_exp/aria/info.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');
selected_AP = info.AP_tag(aria_imdb.selected);
selected_tags = aria_imdb.classes.name(aria_imdb.selected);
[~, sorted_order] = sort(selected_AP, 'descend');
vetted_labels_train = load('/home/phuc/Research/yaromil/vetted_labels_train.mat');
vetted_labels_train = vetted_labels_train.vetted_labels;
vetted_labels_test = load('/home/phuc/Research/yaromil/vetted_labels_test.mat');
vetted_labels_test = vetted_labels_test.vetted_labels;
vetted_labels = [vetted_labels_train vetted_labels_test];

%%
aria_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-390000.mat');
prob = aria_resdb.fc1000.outputs;

%%
vetted_counts = sum(abs(vetted_labels_test) > 1, 2);
[sorted_count, indeces] = sort(vetted_counts, 'descend');
for index = 1:50
  tag_index = indeces(index);
  tag_name = aria_imdb.classes.name{tag_index};
  sc = sorted_count(index);
  tag_prob = prob(tag_index, :);
  [~, sorted_idx] = sort(tag_prob, 'descend');
  vid_indeces = aria_resdb.video_ids(sorted_idx(1:32));
  v_labels = vetted_labels(tag_index, vid_indeces);
  fprintf('%s: vetted count %d, %d out of 32\n', tag_name, sc, sum(abs(v_labels)>1));
end

%% completed tags
tags = {'#cat', '#dance', '#guitar', '#dogs', '#bird',...
  '#horse', '#tennis', '#singing', '#pugsofvine', '#flowers',...
  '#pizza', '#basketball', '#drawings', '#draw', '#kitty', '#clinton',...
  '#volleyball', '#skate', '#snowboarding', '#pugsofvine', '#pikachu',...
  '#kobe', '#butterfly', '#twerk', '#eggs'};
tag_indeces = false(4000, 1);

for tag_index = 1:numel(tags)
  index = find(strcmp(aria_imdb.classes.name, tags{tag_index}));
  tag_indeces(index) = true;
end

true_prec = zeros(numel(tags), 1);
original_prec = zeros(numel(tags), 1);
all_indeces = cell(numel(tags), 1);
v_labels = cell(numel(tags), 1);
o_labels = cell(numel(tags), 1);
cat_indeces = cell(numel(tags), 1); % used this to later index to the right fc1000
for tag_index = 1:numel(tags)
  tag = tags{tag_index};
  cat_index = find(strcmp(aria_imdb.classes.name, tag));
  cat_indeces{tag_index} = repmat(cat_index, [32, 1]);
  cat_prob = prob(cat_index, :);
  [s, indeces] = sort(cat_prob, 'descend');
  vid_indeces = aria_resdb.video_ids(indeces(1:32));
  all_indeces{tag_index} = vid_indeces;
  names = aria_imdb.images.name(vid_indeces);
  v_labels{tag_index} = vetted_labels(cat_index, vid_indeces);
  o_labels{tag_index} = aria_imdb.images.label(cat_index, vid_indeces);
  true_prec(tag_index) = sum(v_labels{tag_index}==2)/32;
  original_prec(tag_index) = sum(o_labels{tag_index}>0)/32;
end

all_indeces = cat(2, all_indeces{:});
v_labels = cat(2, v_labels{:});
o_labels = cat(2, o_labels{:});
cat_indeces = cat(1, cat_indeces{:});

%% train SVM
test_indeces = find(aria_imdb.images.set==2);
count = sum(abs(vetted_labels_test) > 1, 1);
all_vid_indeces = test_indeces(count > 0);
can_used_to_train = setdiff(all_vid_indeces, all_indeces);
can_used_to_test = all_indeces;
can_used_to_train = intersect(can_used_to_train, aria_resdb.video_ids);

%% input ~ fc1000 and observed label, output ~ true label
fc1000 = cell(numel(can_used_to_train), 1);
for i=1:numel(can_used_to_train)
  i
  index_to_prob = find(aria_resdb.video_ids==can_used_to_train(i));
  fc1000{i} = prob(:, index_to_prob);
end
fc1000 = cat(2, fc1000{:});
observed_labels = aria_imdb.images.label(:, can_used_to_train);
true_labels = vetted_labels(:, can_used_to_train);

[a,b] = find(abs(true_labels) > 1);
f = zeros(numel(a), 1);
for i=1:numel(a)
  f(i) = fc1000(a(i), b(i));
end
f = vl_nnsigmoid(f);
training_indeces = find(abs(true_labels) > 1);
y = full(observed_labels(training_indeces) > 0)*2-1;
z = full(true_labels(training_indeces) > 0)*2-1;

model = svmtrain(double(z), double([a f y]), '-c 0.01 -b 1');

%% predicting
num_sample = numel(can_used_to_test);
f_test = zeros(numel(can_used_to_test), 1);
for i=1:numel(can_used_to_test)
  index_to_prob = find(aria_resdb.video_ids==can_used_to_test(i));
  f_test(i) = prob(cat_indeces(i), index_to_prob);
end

observed_labels = aria_imdb.images.label(:, can_used_to_test);
true_labels = vetted_labels(tag_indeces, can_used_to_test);
y_test = zeros(numel(can_used_to_test), 1);
for i=1:numel(can_used_to_test)
  y_test(i) = aria_imdb.images.label(cat_indeces(i), can_used_to_test(i));
end
y_test = (y_test>0)*2 - 1;

z_test = zeros(numel(can_used_to_test), 1);
for i=1:numel(can_used_to_test)
  z_test(i) = vetted_labels(cat_indeces(i), can_used_to_test(i));
end
z_test = (z_test>0)*2 - 1;

[a,b,c] = svmpredict(double(z_test), double([cat_indeces f_test y_test]), model, '-b 1');

%% sampling - random
num_sample = numel(can_used_to_test);
adjusted_prec = zeros(num_sample, 1);
expected_P = zeros(num_sample, 2);
vetted = false(num_sample, 1);
order = randperm(num_sample);
for i=1:num_sample
  vetted(order(i)) = true;
  vetted_part = v_labels(vetted);
  unvetted_part = o_labels(~vetted);
  P_vetted = sum(vetted_part==2);
  P_unvetted = sum(c(~vetted, 1));
  expected_P(i, :) = [P_vetted, P_unvetted];
  adjusted_prec(i) = (P_vetted + sum(unvetted_part==1))/num_sample;
end
expected_P = sum(expected_P, 2)/num_sample;

%% sampling - uncertainty sampling
num_sample = numel(can_used_to_test);
expected_P2 = zeros(num_sample, 2);
vetted = false(num_sample, 1);
[d,order2] = sort((abs(c(:, 1)-.5)));
for i=1:num_sample
  vetted(order2(i)) = true;
  vetted_part = v_labels(vetted);
  unvetted_part = o_labels(~vetted);
  P_vetted = sum(vetted_part==2);
  P_unvetted = sum(c(~vetted, 1));
  expected_P2(i, :) = [P_vetted, P_unvetted];
end
expected_P2 = sum(expected_P2, 2)/num_sample;

%% plotting
close all;
figure(1);
plot([0, num_sample], [mean(true_prec), mean(true_prec)]);
hold on;
plot([0, num_sample], [mean(original_prec), mean(original_prec)]);
plot(0:num_sample, [mean(original_prec); adjusted_prec]);
plot(0:num_sample, [mean(original_prec); expected_P]);
plot(0:num_sample, [mean(original_prec); expected_P2]);
grid on;

figure(2);
plot(1:num_sample, abs(mean(true_prec) - adjusted_prec));
hold on;
plot(1:num_sample, abs(mean(true_prec) - expected_P));
plot(1:num_sample, abs(mean(true_prec) - expected_P2));
grid on;
legend({'adjusted', 'exp+random', 'exp+lc'});