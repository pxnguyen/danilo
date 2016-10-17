% script to run frame extraction
[~,hostname] = system('hostname');
hostname = strtrim(hostname);
opts = struct();
switch hostname
  case 'deepthought'
    matconvnet_path = ''
    opts.dest_dir = '/home/nguyenpx/danilo/vine-images/'
    opts.data_dir = '/home/nguyenpx/vine-large-2/'
    opts.exp_dir = '/home/nguyenpx/danilo/exp_files/'
    opts.num_pool = 12
  case 'pi'
    opts.dest_dir = '/tmp/vine-images/'
    opts.data_dir = '/mnt/large/pxnguyen/vine-large-2/';
    opts.exp_dir = '/mnt/large/pxnguyen/cnn_exp/danilo';
    opts.num_pool = 4;
end

run(fullfile(matconvnet_path, 'matlab', 'vl_setupnn.m'))
cnn_prefetch_frames(opts)
