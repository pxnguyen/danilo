run('/home/phuc/Research/matconvnet-1.0-beta21/matlab/vl_setupnn.m')
metric_layer = dagnn.Metric('top_k', 2);
probs = [...
  .1 .2 .3 .1 .1 .6 .7;
  .3 .1 .1 .9 .6 .8 .2;
  .3 .1 .2 .9 .2 .2 .1;];
labels = [...
  -1 -1  1 -1 -1 -1  1;
   1 -1 -1 -1 -1 -1 -1;
   1 -1 -1 -1 -1  1 -1;
  ];
prec_at_k = metric_layer.forward({probs, labels}, {})
metric_layer

probs = [...
  .5 .2 .8 .1 .1 .6 .7;
  .3 .1 .4 .1 .9 .8 .2;
  .3 .1 .2 .9 .2 .5 .1;];
labels = [...
  -1 -1  1 -1 -1 -1  1;
  -1 -1  1 -1  1  1 -1;
  -1 -1 -1  1  1  1 -1;
  ];

prec_at_k = metric_layer.forward({probs, labels}, {})
metric_layer
