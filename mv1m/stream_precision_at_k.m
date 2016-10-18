function [true_positives, false_positives] = stream_precision_at_k(pred, gt, k)
% Args:
%   pred: the predictions, WxHxCxN
%   gt: the ground truths labels, WxHxCxN
%   pr_type: either per-label or per-image
% N is the number of examples. C is the number of classes
[~, order] = sort(pred, 2, 'descend');
topk_gt = gt(:, order(1:k));
true_positives = sum(sum(topk_gt==1));
false_positives = sum(sum(topk_gt~=1));
