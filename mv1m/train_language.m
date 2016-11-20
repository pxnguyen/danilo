function train_language(tag, imdb)
% train the language model
run('/home/phuc/Research/vlfeat-0.9.20/toolbox/vl_setup.m');
index = find(strcmp(tag, imdb.classes.name));
train = (imdb.images.set==1);
X = imdb.images.label;
Y = full(imdb.images.label(index, :));
Y(Y==0) = -1;
X(index, :) = 0;
X_train = single(full(X(:, train)));
Y_train = double(full(Y(:, train)));
keyboard
svm_model = struct();
[svm_model.w, svm_model.b] = vl_svmtrain(X_train, Y_train, 0.01);
save('language_model.mat', '-struct', 'svm_model');