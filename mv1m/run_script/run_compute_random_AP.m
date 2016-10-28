function run_compute_random_AP
opts = struct();
opts.imdbPath = '/mnt/large/pxnguyen/cnn_exp/danilo/imdb.mat';

fprintf('Loading imdb\n');
tic; imdb = load(opts.imdbPath); toc;

for class_index = 1:imdb.classes.names
end
