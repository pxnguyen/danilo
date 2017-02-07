function print_AP_to_csv(exp_name)
% Write the APs to a csv file
% Args:
%   exp_name: the experiment name (danilo, ari).
% Outputs:
%   None.
exp_dir = fullfile('/mnt/large/pxnguyen/cnn_exp/', exp_name);
info = load(fullfile(exp_dir, 'info.mat'));
imdb = load(fullfile(exp_dir, sprintf('%s_imdb.mat', exp_name)));
fid = fopen(sprintf('%s_ap.csv', exp_name), 'w');
fprintf(fid, 'tag,num_train,AP,dataset\n');
tags = imdb.classes.name;
test = (imdb.images.set==2);
num_positives = sum(imdb.images.label(:, test), 2);
num_total_vid = numel(imdb.images.name(test));
chance_AP = full(num_positives/num_total_vid);
for index = 1:length(tags)
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags{index}, full(info.train_vid_count(index)),...
    info.AP_tag(index), exp_name);
  fprintf(fid, '%s,%0.1f,%0.7f,%s\n',...
    tags{index}, full(info.train_vid_count(index)),...
    chance_AP(index), 'chance');
end
fclose(fid);