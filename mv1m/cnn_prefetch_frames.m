function cnn_prefetch_frames(varargin)
%prefetch and store the frames on ssd

opts.data_dir = '/mnt/large/pxnguyen/vine-large-2/';
opts.exp_dir = '/mnt/large/pxnguyen/cnn_exp/danilo';
opts.dest_dir = '/tmp/vine-images/';
opts.imdbPath = fullfile(opts.exp_dir, 'imdb.mat');
opts.num_pool = 6;
opts = vl_argparse(opts, varargin) ;

imdb = load(opts.imdbPath) ;
imdb.imageDir = fullfile(opts.data_dir, 'videos');

params = opts ;
opts.train = find(imdb.images.set==1);
params.imdb = imdb ;

p = gcp('nocreate'); % If no pool, do not create new one.
if isempty(p)
  parpool('local', opts.num_pool);
end
video_paths = fullfile(imdb.imageDir, imdb.images.name(opts.train));

done = false;
last_check_iter = -1;
parfor video_index = 1:length(video_paths)
  fprintf('%s (%d/%d)\n', video_paths{video_index}, video_index, length(video_paths));
  extract_frames(video_paths{video_index});
end
