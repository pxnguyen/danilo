function cotraining(imdb)
cotag_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria_cotags/resdb-iter-2472000.mat');
vis_resdb = load('/mnt/large/pxnguyen/cnn_exp/aria/resdb-iter-1384000-train.mat');
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');

% PCA down to 512
vis_semantic = vis_resdb.pool5.outputs;
[U,mu,~] = pca( vis_semantic );
%k=946 ~ 90%, 512 - 81%.
tic; [pca_vis_semantic, ~, ~] = pcaApply(vis_semantic, U, mu, 512); toc;

% get the knn
k = 5;

% build the kdtree
tic; kdtree = vl_kdtreebuild(pca_vis_semantic) ; toc;
tic; [idx, ~] = vl_kdtreequery(kdtree, pca_vis_semantic, pca_vis_semantic, 'NumNeighbors', k, 'MaxComparisons', 15000) ; toc;
%tic; Mdl = createns(pca_vis_semantic(:, :)', 'NsMethod'); toc;
%idx = knnsearch(Mdl, pca_vis_semantic', k);

% get the y_hat
y_hat = vl_nnsigmoid(cotag_resdb.fc1000.outputs);

new_imdb_path = '/mnt/large/pxnguyen/cnn_exp/aria_cotraining/aria_cotraining_imdb.mat';
new_imdb = aria_imdb;
new_imdb.closest_neighbors = idx;
save(new_imdb_path, '-struct', 'new_imdb');

latent_labels_path = '/mnt/large/pxnguyen/cnn_exp/aria_cotraining/latent_labels.mat';
latent_labels = struct();
latent_labels.value = aria_imdb.images.label;
latent_labels.value(:, cotag_resdb.video_ids) = y_hat;
save(latent_labels_path, '-v7.3', '-struct', 'latent_labels');