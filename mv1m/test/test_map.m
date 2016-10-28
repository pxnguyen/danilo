gts = [0 1 0 0;
       1 1 0 0];
preds = [.8 .1 .2 .5;
         .1 .5 .2 .7];

compute_average_precision(preds, gts)
