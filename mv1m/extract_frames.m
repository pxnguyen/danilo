function files = extract_frames(vid_path, varargin)
% Given a video path, extract the frames into a tmp folder
%
% If the number of r
%
% Args:
%   fps: the frame rate to extract frames
%   imageSize: the image size of the extracted frames
%   force: whether to delete the folder
%   dest_dir: where to write the frames
%   min_num_frame: the minimum number of frames to get from this videos
%   max_num_retry: maximum number of retries to extract frames

opts.fps = 5;
opts.imageSize = [256, 256];
opts.force = false;
opts.dest_dir = '/tmp/vine-images/';
opts.min_num_frame = 15;
opts.max_num_retry = 5;
opts = vl_argparse(opts, varargin);

if ~exist(vid_path, 'file')
  msgID = 'extractFrames:FILENOTFOUND';
  msg = sprintf('Can not find %s', vid_path);
  baseException = MException(msgID, msg);
  throw(baseException)
end

[~, name, ext] = fileparts(vid_path);
frame_tmpdir = fullfile(opts.dest_dir, [name ext]);

if ~opts.force
  if exist(frame_tmpdir, 'dir') && length(dir(fullfile(frame_tmpdir, '*.jpeg'))) > opts.min_num_frame
    files = dir(fullfile(frame_tmpdir, '*.jpeg'));
    files = strcat([frame_tmpdir filesep], {files.name});
    return;
  end
end

if ~exist(frame_tmpdir, 'dir') %  make the folder if it doesn't exist
  mkdir(frame_tmpdir);
end;

done = false;
num_tries = 0;
while ~done
  [~,hostname] = system('hostname');
  hostname = strtrim(hostname);
  if strcmp(hostname, 'pi') || strcmp(hostname, 'delta')
    cmd = sprintf('avconv -i "%s" -s %dx%d -r %d %s/image-%%3d.jpeg',...
      vid_path, opts.imageSize(2), opts.imageSize(1), opts.fps, frame_tmpdir);
  else
    cmd = sprintf('ffmpeg -i "%s" -s %dx%d -r %d %s/image-%%3d.jpeg',...
      vid_path, opts.imageSize(2), opts.imageSize(1), opts.fps, frame_tmpdir);
  end
  [~, cmdout] = unix(cmd);

  frame_count = length(dir(fullfile(frame_tmpdir, '*.jpeg')));
  if frame_count > opts.min_num_frame % good to go
    done = true;
  else
    num_tries = num_tries + 1;
    if num_tries > opts.max_num_retry
      done = true;
      if frame_count > 0 % if we are able to get some frames, replicate these
        files = dir(fullfile(frame_tmpdir, '*.jpeg'));
        for i=1:opts.min_num_frame - frame_count + 5
          from_file = fullfile(opts.dest_dir, [name ext], files(randi(frame_count)).name);
          to_file = fullfile(opts.dest_dir, [name ext], sprintf('image-%03d.jpeg', frame_count+i));
          copyfile(from_file, to_file);
        end
      else % return random images
        fprintf('Outputing random images\nq');
        fwrite(fid, sprintf('%s\n', [name ext]));
        for i=1:length(opts.min_num_frame) + 5
          random_image = uint8(randi(255, [opts.imageSize 3]));
          to_file = fullfile(opts.dest_dir, [name ext], sprintf('image-%03d.jpeg', i));
          imwrite(random_image, to_file);
        end
      end
    end
  end
end

files = dir(fullfile(frame_tmpdir, '*.jpeg'));
files = strcat([frame_tmpdir filesep], {files.name});
