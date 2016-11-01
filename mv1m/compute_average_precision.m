function AP_tag = compute_average_precision(preds, gts, varargin)
% Compute the average precision.
% Args:
%   preds: the output probabibilty, num_classes x num_examples
%   gts: the ground truth labels, num_classes x num_examples
%   opts.aggregate_type: per-tag or per-example
%   opts.
[C, N] = size(preds);
[Cg, Ng] = size(preds);
pos_num_class = [4000 1000 807 40 51 15 76 4 2];
if (~any(ismember(pos_num_class, C))) || C ~= Cg || N~=Ng;
  throw(MException('MYFUN:BadIndex', 'wrong inputs'));
end
opts.aggregate_type = 'per-class';
opts = vl_argparse(opts, varargin);

AP_tag = zeros(C, 1);
for class_index = 1:C
  fprintf('%d/%d\n', class_index, C);
  preds_class = single(preds(class_index, :));
  gts_class = single(gts(class_index, :));
  AP_tag(class_index) = compute_ap_class(preds_class, gts_class);
end

function ap = compute_ap_class(preds, gts)
if isempty(preds)
  ap = 0
  return
end
[~, N] = size(preds);
[~, order] = sort(preds, 2, 'descend');
relevant = gts(order);
cumsum_rel = cumsum(relevant);
alpha = (1:N);
val = cumsum_rel .* relevant ./alpha;
ap = sum(val)./ (sum(relevant));
