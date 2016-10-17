function [averageImage, rgbMean, rgbCovariance] = getImageStats(images, varargin)
%GETIMAGESTATS  Get image statistics

opts.gpus = [] ;
opts.batchSize = 12 ;
opts.imageSize = [256 256] ;
opts.numThreads = 6 ;
opts = vl_argparse(opts, varargin) ;

avg = {} ;
rgbm1 = {} ;
rgbm2 = {} ;

numGpus = numel(opts.gpus) ;
if numGpus > 0
  fprintf('%s: resetting GPU device\n', mfilename) ;
  clear mex ;
  gpuDevice(opts.gpus(1))
end

for t=1:opts.batchSize:numel(images)
  time = tic ;
  batch = t : min(t+opts.batchSize-1, numel(images)) ;
  fprintf('collecting image stats: batch starting with image %d ...', batch(1)) ;
  
  video_paths = images(batch);
  all_files = cell(length(video_paths),1);
  for video_index = 1:length(video_paths)
    all_files{video_index} = extract_frames(video_paths{video_index});
  end
  
  all_files = cat(2, all_files{:});
  
  data = getImageBatch(all_files, ...
                       'numThreads', opts.numThreads, ...
                       'imageSize', opts.imageSize, ...
                       'useGpu', numGpus > 0) ;

  z = reshape(shiftdim(data,2),3,[]) ;
  rgbm1{end+1} = mean(z,2) ;
  rgbm2{end+1} = z*z'/size(z,2) ;
  avg{end+1} = mean(data, 4) ;
  time = toc(time) ;
  fprintf(' %.1f Hz\n', numel(batch) / time) ;
end

averageImage = gather(mean(cat(4,avg{:}),4)) ;
rgbm1 = gather(mean(cat(2,rgbm1{:}),2)) ;
rgbm2 = gather(mean(cat(3,rgbm2{:}),3)) ;
rgbMean = rgbm1 ;
rgbCovariance = rgbm2 - rgbm1*rgbm1' ;

if numGpus > 0
  fprintf('%s: finished with GPU device, resetting again\n', mfilename) ;
  clear mex ;
  gpuDevice(opts.gpus(1)) ;
end
fprintf('%s: all done\n', mfilename) ;
