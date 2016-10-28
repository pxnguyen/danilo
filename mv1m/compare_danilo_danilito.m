function compare_danilo_danilito
danilo_dir = '/mnt/large/pxnguyen/cnn_exp/danilo/';
danilito_dir = '/mnt/large/pxnguyen/cnn_exp/danilito/';
danilo_info = load(fullfile(danilo_dir, 'info.mat'));
danilito_info = load(fullfile(danilito_dir, 'info.mat'));
danilo_AP_tag = danilo_info.AP_tag;
danilito_AP_tag = danilito_info.AP_tag;
danilito_vid_count = full(danilito_info.train_vid_count);
imdb = load(fullfile(danilo_dir, 'imdb.mat'));
train = (imdb.images.set == 1);
vid_count = full(sum(imdb.images.label(:, train), 2));
eval_indeces = (imdb.images.set == 2);
N = sum(eval_indeces);
%fid = fopen('ap.csv', 'w');
%fprintf(fid, 'Tag Name,num_train,AP\n');
%fprintf(fid, 'Tag Name,danilo_num_tr,danilo_ap,danilito_num_tr,danilito_ap\n');
aggregate_results = struct();
aggregate_results.tag_names = imdb.classes.name;
aggregate_results.danilito_num_tr = danilito_vid_count;
aggregate_results.danilo_num_tr = vid_count;
aggregate_results.danilo_AP = danilo_AP_tag;
aggregate_results.danilito_AP = danilito_AP_tag;
aggregate_results.chance_AP = zeros(numel(imdb.classes.name), 1);
fprintf('Computing the chance AP...\n');
for index = 1:length(imdb.classes.name)
  class_indeces = imdb.images.label(index, :);
  eval_vid_indeces = find(class_indeces & eval_indeces);
  num_positives = numel(eval_vid_indeces);
  aggregate_results.chance_AP(index) = single(num_positives)/N;
end
fprintf('Saving the results...\n');
save_path = '/mnt/large/pxnguyen/cnn_exp/danilo/aggregate_results.mat';
save(save_path, '-struct', 'aggregate_results');
%   comparisons.tag_name(index) = imdb.classes.name{index};
%   danilo_tr_vid = vid_count(index);
%   danilo_AP = danilo_AP_tag(index);
%   danilito_AP = danilito_AP_tag(index);
%   if isnan(danilo_AP)
%     danilo_AP = 0.0000001;
%   end
%   danilo_AP = max(0.000001, danilo_AP);
%   danilito_tr_vid = danilito_vid_count(index);
%   danilito_AP = danilito_AP_tag(index);
%   %fprintf(fid, sprintf('%s,%0.1f,%0.7f\n',...
%   %  tag_name, danilo_tr_vid, danilo_AP));
%   if danilo_AP > chance_AP && danilo_AP > danilito_AP
%     %fprintf('%s,%0.6f,%0.6f,%0.2f,%0.6f\n',...
%     %  tag_name, danilo_AP, danilito_AP, (danilo_AP/danilito_AP - 1)*100, chance_AP);
%     improvement(index) = (danilo_AP/danilito_AP - 1)*100;
%   end
%fclose(fid);
