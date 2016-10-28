run('/home/phuc/Research/danilo/mv1m/matconvnet/matlab/vl_setupnn.m')
probs = [...
  .1 .2 .3 .1 .1 .6 .7;
  .3 .1 .1 .9 .6 .8 .2;
  .3 .1 .5 .4 .2 .2 .1;];
probs = permute(probs, [3 4 2 1]);
c = [...
  -1 -1  1 -1 -1  1 -1;
   1 -1 -1  1 -1  1 -1;
   1 -1  1 -1 -1  1 -1;
  ];
c = permute(c, [3 4 2 1]);
vl_nnloss(probs, c, [], 'loss', 'hit@k', 'topK', 1)
vl_nnloss(probs, c, [], 'loss', 'hit@k', 'topK', 2)
