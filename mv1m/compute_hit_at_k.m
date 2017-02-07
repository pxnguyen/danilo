function prec = compute_precision_at_k(preds, gts, varargin)
% Compute the average precision.
% Args:
%   preds: the output probabibilty, num_classes x num_examples
%   gts: the ground truth labels, num_classes x num_examples
%   opts.aggregate_type: per-tag or per-example
% test case:
[C, N] = size(preds);
[Cg, Ng] = size(gts);
pos_num_class = [4000 1000 807 40 51 15 76 4 2 1];
% if (~any(ismember(pos_num_class, C))) || C ~= Cg || N~=Ng;
%   throw(MException('MYFUN:BadIndex', 'wrong inputs'));
% end
opts.aggregate_type = 'per-class';
opts.k = 10;
opts = vl_argparse(opts, varargin);

prec = zeros(C, 1);
for class_index = 1:C
  class_index
  preds_class = single(preds(class_index, :));
  gts_class = single(gts(class_index, :));
  prec(class_index) = compute_prec_at_k(preds_class, gts_class, opts.k);
end

function prec = compute_prec_at_k(preds, gts, k)
if isempty(preds)
  ap = 0;
  return
end
[~, N] = size(preds);
[~, order] = sort(preds, 2, 'descend');
order = order(1:k);
relevant = gts(order);
prec = sum(relevant) / k;
