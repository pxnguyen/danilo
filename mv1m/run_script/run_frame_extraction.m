function run_frame_extraction(imdb, dataset_name)
% script to run frame extraction
[~,hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'deepthought'
    matconvnet_path = '/home/nguyenpx/matconvnet/';
    opts.dest_dir = '/home/nguyenpx/danilo/vine-images/';
    opts.data_dir = '/home/nguyenpx/vine-large-2/';
    opts.exp_dir = '/home/nguyenpx/danilo/exp_files/';
    opts.num_pool = 12;
  case 'pi'
    matconvnet_path = '/home/phuc/Research/matconvnet-1.0-beta21/';
    opts.dest_dir = '/mnt/hermes/nguyenpx/vine-images/';
    opts.data_dir = imdb.imageDir;
    opts.exp_dir = sprintf('/mnt/large/pxnguyen/cnn_exp/%s', dataset_name);
    opts.dataset_name = dataset_name;
    opts.num_pool = 4;
end

run(fullfile(matconvnet_path, 'matlab', 'vl_setupnn.m'))
cnn_prefetch_frames(opts)