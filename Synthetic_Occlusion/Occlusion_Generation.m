%% SYNTHETIC OCCLUSION GENERATOR

%Code is split in sections so that the user can run the sections he needs.

%CODE STRUCTURE
% 1. Load Occlusions (.mat file);
% 2. Load Original Dataset to be occluded;
% 2.1. Random Fixed Occlusions Indices (optional)
% 3. Determining Threshold Distance for First Frame
% 3.1. generate output cluster data (optional)
% 3.2. Plot result cluster (optional)
% 4. Iterate and Generate New Synthetically Occluded Dataset
%% 1. Load Occlusion .mat File 

load('occlusion_1.mat');
Fixed_Occlusion_Flag = 0; %flag to indicate we the user desires fixed occ.

%% 2. Load Images in Dataset 

%images = dir('Occluded_faces\*.jpg');
images = dir('D:\datasets\300W_LP\AFW\*.jpg')
%images = dir('D:\datasets\faces_biwi_wide\*.jpg')
images = natsortfiles(images);

% % Define a starting folder.
% start_path = fullfile(matlabroot, '\toolbox\images\imdemos');
% % Ask user to confirm or change.
% topLevelFolder = uigetdir(start_path);
% if topLevelFolder == 0
% 	return;
% end
% 
% % Get list of all subfolders.
% allSubFolders = genpath(topLevelFolder);
% % Parse into a cell array.
% remain = allSubFolders;
% listOfFolderNames = {};
% while true
% 	[singleSubFolder, remain] = strtok(remain, ';');
% 	if isempty(singleSubFolder)
% 		break;
% 	end
% 	listOfFolderNames = [listOfFolderNames singleSubFolder];
% end
% numberOfFolders = length(listOfFolderNames)


% %% 2.1. Random Fixed Occlusions Indices (optional)
% % If you desire to apply the same random occlusions to more than one
% % dataset run this section.
% 
% Fixed_Occlusion_Flag = 1; %flag to indicate we the user desires fixed occ.
% matrix_index = zeros(length(images),1);
% 
% % A fixed random occlusion for each dataset image
% for i = 1:length(images)
%     
%     %first frame from .mat file must be non-occluded, we desire index from
%     %2 to the number of occlusions.
%     ind = randperm(length(points)-1,1)+1; 
%     matrix_index(i,:) = ind;
%     
% end


%% 3. Determining Threshold Distance for First Frame

% Initialize structure to store occluded images if necessary.
%occluded_images = cell(1, length(images));

% Face Detection (no occlusion)
rgb1 = RGB{1};
xyz1_plot = Depth{1};
xyz1 = double(reshape(xyz1_plot, [480*640 1]))'; 

%image(rgb1)
%image(xyz1)
% Apply face detector to image                                                       
faceDetector = vision.CascadeObjectDetector();
rect = step(faceDetector, rgb1);

% Draw the returned bounding box around the detected face.
%rgb1= insertShape(rgb1, 'Rectangle', rect);
%xyz1= insertShape(xyz1, 'Rectangle', rect);

%figure; imshow(rgb1); title('Detected face');
%figure; imshow(xyz1); title('Detected face depth');


% Limits of Bounding Box(rect)
ymin = rect(2);
ymax = rect(2) + rect(4);
xmin = rect(1);
xmax = rect(1) + rect(3);

% If you desire to broaden or reduce the border of the face detector,
% uncomment the lines below
%k = 0.2;
% x_min = xmin - k * abs(xmax - xmin);
% y_min = ymin - k * abs(ymax - ymin);
% x_max = xmax + k * abs(xmax - xmin);
% y_max = ymax + k * abs(ymax - ymin);

%if you desire to use the bounding box of the face detector
x_min = xmin;
y_min = ymin;
x_max = xmax;
y_max = ymax;

rect_new = [x_min, y_min, x_max-x_min, y_max-y_min]; 

im_virtual = zeros(480,640);
im_virtual(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max)) = 1;
%im_virtual(ceil(rect(2)):ceil(rect(2))+ceil(rect(4)), ceil(rect(1)):ceil(rect(1))+ceil(rect(3))) = 1; %puts face box in with 1s
im_virtual = im_virtual .* double(rgb2gray(rgb1));
face_box_index = find(im_virtual ~= 0);
orig_face_xyz = xyz1(:,face_box_index); orig_face_xyz = orig_face_xyz'; 

% % CLUSTERING TO AVOID OUTLIERS IN DISTANCE SEPARATION
%addpath('utils');
%[class,~] = dbscan(double(orig_face_xyz), 500, 20);
[class,~] = dbscan(orig_face_xyz,5,200);
maxClass = max(class)
%biggest cluster will correspond to the face 
for j=1:max(class)
    tam = size(find(class == j));
    %counter(j) = tam(1,1)
    counter(j) = tam(1);
end

% find points of biggest cluster (face cluster)
big_class = find(counter == max(counter));
cara_index = find (class == big_class);
%face = orig_face_xyz(cara_index); % face xyz points

face = orig_face_xyz;
% Threshold distance that identifies occlusions
face(face==0)=inf; %remove zeros from matrix
distance_sep = min(face(:,1));

% Threshold distance that identifies occlusions
%distance_sep = min(face)


%% 4. Iterate and Generate New Synthetically Occluded Dataset

% For each dataset image
for z = 1:length(images) % or length(images) for whole dataset
    image_data = images(z);
    image_data = imread(fullfile(image_data.folder,image_data.name));
    
    
    if Fixed_Occlusion_Flag == 1 %If fixed occlusions are desired
        i = matrix_index(z,1);
    else                            
        i = randperm(length(Depth)-1,1)+1; %select random occlusion
    end
    
    
    %depth image
    xyz = Depth{i};
    
    %color image
    rgb = RGB{i};
    
    %crop rgb and depth within detection box
    frame_bbox = rgb(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max),1:3);
    face_rgb = reshape(frame_bbox, [size(frame_bbox,1)*size(frame_bbox,2) 3]);
    face_xyz = xyz(ceil(y_min):ceil(y_max),ceil(x_min):ceil(x_max)); 
    face_xyz = reshape(face_xyz, [size(face_xyz,1)*size(face_xyz,2) 1]);
    
    face_xyz(face_xyz==0)=inf; %remove zeros from matrix

    
    %select points closer than threshold distance (occlusion points)
    valid = face_xyz(:,1) < distance_sep * 1.2; 
    face_xyz = face_xyz(valid,:);

    

    
    
    %crop image to face box

    

    
    %we only want the obstacle to overlay on the dataset image
    face_rgb(~valid,:) = 0;
    face_xyz(~valid,:) = 0;
    %reshape vector to face box dimension
    occlusion = reshape(face_rgb, [size(frame_bbox,1), size(frame_bbox,2), 3]);
    
    %occlusion_depth = reshape(face_xyz, [size(frame_bbox,1), size(frame_bbox,2), 1]);
    %figure; imshow(occlusion)
    %figure; surf(occlusion_depth)
    
    %writes image in destiny folder
    folder_object = 'occluded_objects\';
    FILENAME = string(strcat(folder_object , 'object', num2str(i), '.jpg'));
    imwrite(occlusion,FILENAME);
    
    %rescales object image size to dataset image size
    K = 0.7;
    temp = imresize(occlusion,[size(image_data,1)*K,size(image_data,2)*K]); 
    s = size(temp);
    
    occlusion_resize = zeros(size(image_data,1),size(image_data,2),3); % all zero matrix
    occlusion_resize(1:s(1), 1:s(2), :) = temp; % oldMatrix is a, newMatrix is b
    %socclusion_resize = imresize(occlusion_resize, 0.7);
    
    
    
    
   
    
    
    
    imcropped_vector = reshape(image_data,[size(image_data,1)*size(image_data,2),3]); %matrix of dataset image
    occlusion_resize_vector = reshape(occlusion_resize,[size(image_data,1)*size(image_data,2),3]); %matrix of rescaled object image
   
  
    %finds nonzero elements(image occlusion pixels);
    occlusion_index = find(occlusion_resize_vector(:,1)); 
    %superimposes occlusion in original dataset image
    imcropped_vector(occlusion_index,:) = occlusion_resize_vector(occlusion_index,:);
    
    %reshapes synthetic occluded image matrix to original image shape
    data_occluded = reshape(imcropped_vector,[size(image_data,1),size(image_data,2),3]);
      
    if ~(mod(z,1000))
        fprintf('Occlusion %d \n',z);
    end
    
    %writes image in destiny folder
    folder = 'Destiny_Folder\';
    %folder = 'D:\datasets\biwi_wide_feedbot_occluded\'
    FILENAME = string(strcat(folder, 'oclusion', num2str(z), '.jpg'));
    imwrite(data_occluded,FILENAME);
    
end