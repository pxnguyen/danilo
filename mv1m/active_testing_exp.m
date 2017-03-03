  function active_testing_exp
gt = [1 0 0 1 0];
pred = [0.5 0.1 0.2 0.4 0.9];
adjusted = [0.7 0.02 0.1 0.5 0.4];
m = mu(gt, adjusted);
u = ub(gt, adjusted);
l = lb(gt, adjusted);
dub = u-m;
dlb = m-l;
fprintf('%0.2f %0.2f %0.2f %0.2f %0.2f\n', m, u, l, u-m, m-l);

for i=1:5
  % stay
  gt2 = gt;
  adj2 = adjusted;
  x = gt2(i);
  gt2(i) = [];
  adj2(i) = [];
  m = x+mu(gt2, adj2);
  u = x+ub(gt2, adj2);
  l = x+lb(gt2, adj2);
  fprintf('%d stay: %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n', i, u, m, l, u-m, m-l, u-l);
  
  % flip
  gt2 = gt;
  adj2 = adjusted;
  x = ~gt2(i);
  gt2(i) = [];
  adj2(i) = [];
  m = x+mu(gt2, adj2);
  u = x+ub(gt2, adj2);
  l = x+lb(gt2, adj2);
  fprintf('%d flip: %0.2f %0.2f %0.2f %0.2f %0.2f %0.2f\n', i, u, m, l, u-m, m-l, u-l);
end

function m=mu(gt, adjusted)
m = sum(adjusted);

function u=ub(gt, adjusted)
u = sum(gt + (~gt).*adjusted);

function l=lb(gt, adjusted)
l = sum(gt.*adjusted);