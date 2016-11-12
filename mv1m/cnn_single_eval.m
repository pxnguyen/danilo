function cnn_single_eval(video_path, net)
root_exp_dir = '/mnt/large/pxnguyen/cnn_exp';
videoDir = '/mnt/large/pxnguyen/vine-lage-2/videos';
imdb = load(fullfile(root_exp_dir, 'ari_nospam_small', 'ari_nospam_small_imdb.mat'));
imageStatsPath = fullfile(root_exp_dir, 'ari_nospam_small', 'imageStats.mat');

% load the image stats
load(imageStatsPath, 'averageImage', 'rgbMean', 'rgbCovariance') ;
[v,d] = eig(rgbCovariance) ;
rgbDeviation = v*sqrt(d) ;
clear v d ;

meta = net.meta;
if numel(meta.normalization.averageImage) == 3
  mu = double(meta.normalization.averageImage(:)) ;
else
  mu = imresize(single(meta.normalization.averageImage), ...
    meta.normalization.imageSize(1:2)) ;
end

frame_dir = '/mnt/hermes/nguyenpx/vine-images/';
useGpu = true;
all_files = extract_frames(video_path, 'dest_dir', frame_dir);
frame_selection = floor(linspace(1, length(all_files), 5));
all_files = all_files(frame_selection);

bopts.train = struct(...
  'useGpu', useGpu, ...
  'numThreads', 4, ...
  'imageSize',  meta.normalization.imageSize(1:2), ...
  'cropSize', meta.normalization.cropSize, ...
  'subtractAverage', mu) ;

% Copy the parameters for data augmentation
for f = fieldnames(meta.augmentation)'
  f = char(f) ;
  bopts.train.(f) = meta.augmentation.(f) ;
end

data = getImageBatch(all_files, bopts.train) ;
labels = single(ones(807, 1));
labels = permute(labels, [3, 4, 1, 2]);
inputs = {'input', data, 'label', labels} ;

% save the sigmoid layers
sel = find(cellfun(@(x) strcmp(x, 'sigmoid'), {net.vars.name})) ;
net.vars(sel).precious = 1;

net.meta.curBatchSize = 1;

net = dagnn.DagNN.loadobj(net) ;
net.mode = 'normal' ;
net.move('gpu') ;
net.eval(inputs) ; % forward pass

sigmoid = gather(permute(net.vars(sel).value, [3 4 1 2]));
[prob_sorted, sorted] = sort(sigmoid, 'descend');
for i=1:10
  fprintf('%s:%0.3f\n', imdb.classes.name{sorted(i)}, prob_sorted(i)*100);
end
