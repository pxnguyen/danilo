function info=cnn_compute_mAP(varargin)
opts = struct();
opts.expDir = '/mnt/large/pxnguyen/cnn_exp/danilo';
opts.resdb_path = fullfile(opts.expDir, 'resdb-iter-69000.mat');
opts.imdbPath = '';
opts = vl_argparse(opts, varargin)

fprintf('Loading imdb\n');
tic; imdb = load(opts.imdbPath); toc;

info_path = fullfile(opts.expDir, 'info.mat');
fprintf('Loading resdb\n');
tic; resdb = load(opts.resdb_path); toc;
fprintf('Computing APs...\n');
train_indeces = find(imdb.images.set==1);
info.train_vid_count = sum(imdb.images.label(:, train_indeces), 2);
info.AP_tag = compute_average_precision(resdb.fc1000.outputs, resdb.gts);

function print_to_google_vis(vid_count, AP_tag, names)
for index = 1:length(names)
  fprintf('[%d, %0.3f, "%s"], ', vid_count(index), AP_tag(index),...
    names{index});
end

function draw_scatter_plot(tag_count, AP_tag)
semilogx(tag_count, AP_tag, 'x');
grid on;
% draw the scatter plot
%addpath(genpath('MexConv3D'))
%opts = struct();
%opts.train = struct();
%opts.train.gpus = [1];
%opts.imdbPath = '/mnt/large/pxnguyen/cnn_exp/danilo/imdb.mat';
%cnn_mv1m_evaluate(opts);
