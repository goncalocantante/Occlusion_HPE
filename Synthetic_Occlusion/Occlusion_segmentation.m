close all
clear all

filename= 'rosbagFiles/2021-12-23-17-53-20';
bag=rosbag(strcat(filename,'.bag'));

duration = bag.EndTime-bag.StartTime;
ii = duration * 0.174;
frameIdx = 1;

%%
% time_ = [bag.StartTime+ii bag.StartTime+ii+1];
% bag_rgb = select(bag, 'Time', time_, 'Topic', '/camera/color/image_raw');
% msgs_rgb = readMessages(bag_rgb);
% bag_depth = select(bag, 'Time', time_, 'Topic', '/camera/aligned_depth_to_color/image_raw');
% msgs_depth = readMessages(bag_depth);
% 
% time_ = [bag.StartTime+ii bag.StartsTime+ii+1];
% bag_rgb = select(bag, 'Time', time_, 'Topic', '/camera/color/image_raw');
% [depthImage,] = readImage(msgs_depth{1}); % Capture Depth image.
% [colorImage,] = readImage(msgs_rgb{1}); % Capture RGB image.
% 
% imshow(colorImage)

%%
while ii+1< duration*0.24,
    time_ = [bag.StartTime+ii bag.StartTime+ii+1];
    bag_rgb = select(bag, 'Time', time_, 'Topic', '/camera/color/image_raw');
    msgs_rgb = readMessages(bag_rgb);
    bag_depth = select(bag, 'Time', time_, 'Topic', '/camera/aligned_depth_to_color/image_raw');
    msgs_depth = readMessages(bag_depth);
        
    for jj = 1:length(bag_rgb),
        [colorImage,] = readImage(msgs_rgb{jj}); % Capture RGB image.
        [depthImage,] = readImage(msgs_depth{jj}); % Capture Depth image.
        %converter para metros a depth (está em mm)
        
        % Store all data for respective frame.
        RGB{frameIdx} = colorImage;
        Depth{frameIdx} = depthImage;
        %Depth{frameIdx} = double(reshape(depthImage, [480*640 1]))';
       
        frameIdx = frameIdx + 1;
        
        %figure; imshow(colorImage) 
       
    end 
    ii = ii + 1;
end

% %%
% aa = 1;
% [numRows,numCols] = size(Depth)
% while aa < numCols + 1
%     figure;
% %     depthImage = cell2mat(Depth(1, aa));
%     %surf(depthImage)
%     imshow(RGB(aa))
%     aa = aa+1;
% end

%% SAVE THE DATA IN .MAT FILE

Filename = 'occlusion.mat'; 
save(Filename, 'RGB', 'Depth');
%save(Filename, 'points', 'RGB', 'Depth');
