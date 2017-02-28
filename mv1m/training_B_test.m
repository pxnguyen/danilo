%% this script is to test the implementation of training B.
aria_imdb = load('/mnt/large/pxnguyen/cnn_exp/aria/aria_imdb.mat');

%% run test
names = {'a', 'b', 'c', 'd'};
net = cnn_mv1m_init_language_model('features', {'test'}, 'classNames', names);
%%
for i=1:numel(net.vars)
  net.vars(i).precious = 1;
end
observed_input = single([1 0 1 1; 1 1 1 0;]);
observed_input = permute(observed_input, [3, 4, 2, 1]);
corrupted_input = single([1 0 0 0; 1 1 0 0;]);
corrupted_input = permute(corrupted_input, [3, 4, 2, 1]);
latent_label = single([0.5 0.1 0.1 0.7; 0.5 0.1 0.1 0.7; ]);
latent_label = permute(latent_label, [3, 4, 2, 1]);
inputs = {'corrupted_input', corrupted_input,...
  'observed_input', observed_input,...
  'latent_label', latent_label};
net.eval(inputs, {'loss1', 0, 'loss2', 1})

%% checking
% vars
% 1 - observer_input
% 2 - fc1
% 3 - corrupted_input
% 4 - fc2
% 5 - latent_label
% 6 - loss1
% 7 - loss2

l1 = net.vars(6).value
l2 = net.vars(7).value
fc1 = net.vars(2).value
fc2 = net.vars(4).value