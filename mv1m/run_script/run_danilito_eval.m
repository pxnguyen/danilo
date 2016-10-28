function run_danilito_eval
run('/home/phuc/Research/danilo/mv1m/matconvnet/matlab/vl_setupnn.m');
opts = struct();
opts.expDir = '/mnt/large/pxnguyen/cnn_exp/danilito';
opts.resdb_path = fullfile(opts.expDir, 'resdb-iter-81000.mat');
opts.imdbPath = fullfile(opts.expDir, 'danilito_imdb.mat');

cnn_compute_mAP(opts);

function resdb = combine_resdb(resDir)
%TODO(phuc): move this into cnn_eval_dag
list = dir(fullfile(resDir, 'resdb-iter-*.temp-*.mat'));
resdb = struct();
resdb.names = cell(length(list), 1);
resdb.predictions = cell(length(list), 1);
resdb.groundtruths = cell(length(list), 1);
for list_index = 1:length(list)
  fprintf('%d/%d\n', list_index, length(list));
  part_resdb = load(fullfile(resDir, list(list_index).name));
  part_resdb.predictions = cat(2, part_resdb.predictions{:});
  resdb.names{list_index} = part_resdb.name;
  resdb.predictions{list_index} = part_resdb.predictions;
  resdb.groundtruths{list_index} = part_resdb.groundtruths;
  clear part_resdb
end

resdb.names = cat(2, resdb.names{:});
resdb.predictions = cat(2, resdb.predictions{:});
resdb.groundtruths = cat(2, resdb.groundtruths{:});
