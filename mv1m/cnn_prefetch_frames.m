function cnn_prefetch_frames(varargin)
%prefetch and store the frames on ssd

run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;

opts.dataDir = '/mnt/large/pxnguyen/vine-large-2/';
opts.expDir = fullfile('/mnt/large/pxnguyen/cnn_exp/danilo');
opts.dest_dir = '/tmp/vine-images/';
opts.imdbPath = fullfile(opts.expDir, 'imdb.mat');
opts = vl_argparse(opts, varargin) ;

imdb = load(opts.imdbPath) ;
imdb.imageDir = fullfile(opts.dataDir, 'videos');

params = opts ;
opts.train = find(imdb.images.set==1);
params.imdb = imdb ;

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
  parpool('local', 4);
end

done = false;
last_check_iter = -1;
while ~done
  [epoch, iter] = findLastCheckpoint(opts.expDir);
  if last_check_iter ~= iter
    info = load(fullfile(opts.expDir, 'train_random_order.mat'));
    video_order = info.train_random_order;
    train_indeces = opts.train(video_order) ; % shuffle
    video_paths = fullfile(imdb.imageDir, imdb.images.name(train_indeces));
    batchSize = info.batchSize;
    max_iter = floor(numel(video_order)/batchSize);
    start_video = batchSize*mod(iter, max_iter);
    last_check_iter = iter;
    fprintf('Loading from checkpoint at epoch %d iter %d\n', epoch, iter);
    % remove the past
    %for video_index = 1:start_video
    %  vid_path = video_paths{video_index};
    %  [~, name, ext] = fileparts(vid_path);
    %  frame_tmpdir = fullfile(opts.dest_dir, [name ext]);
    %  if exist(frame_tmpdir, 'dir')
    %    fprintf('removing %s (%d/%d)\n', frame_tmpdir, video_index, start_video)
    %    rmdir(frame_tmpdir, 's');
    %  end
    %end

    parfor video_index = start_video+1:min(start_video+2000*batchSize, length(video_paths))
      fprintf('%s (%d/%d)\n', video_paths{video_index}, video_index, length(video_paths));
      extract_frames(video_paths{video_index});
    end
  end

  fprintf('Pausing for 60 seconds...\n');
  pause(60);
end

% -------------------------------------------------------------------------
function [epoch, iter] = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*-iter-*.mat')) ;
if ~isempty(list)
  tokens = regexp({list.name}, 'net-epoch-([\d]+)-iter-([\d]+).mat', 'tokens') ;
  epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
  iter = cellfun(@(x) sscanf(x{1}{2}, '%d'), tokens) ;
  epoch = max([epoch 0]);
  iter = max([iter 0]);
else
  epoch = 1;
  iter = 0;
end
