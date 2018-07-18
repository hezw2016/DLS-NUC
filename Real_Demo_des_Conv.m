% close all;
clear all;
run matconvnet-1.0-beta24/matlab/vl_setupnn;
addpath('utils')


load('./desCNN_Models/model1.mat');v=4; %change to model2

folder = './Data/thermal_test';
% folder = './Data/mytest';


use_gpu = 1;
index = size(model.weight,2);
if use_gpu
    for i = 1:index
        model.weight{i} = gpuArray(model.weight{i});
        model.bias{i} = gpuArray(model.bias{i});
    end
end

switch v
    case 1
        scale = 3;
    case 2
        scale = 3;
    case 3
        scale = 4;
    case 4
        scale = 3;
end


filepaths = dir(fullfile(folder,'*.png'));
% for i = 1:length(filepaths) %运行所有图片
for i = 3 : 3
fprintf('%d / %d \n',i,length(filepaths));

im_noise = imread(fullfile(folder,filepaths(i).name)); 

im_noise = modcrop(im_noise,12);%%模型下采样为scale倍

if size(im_noise, 3) == 3
    im_noise = rgb2gray(im_noise);
end
im_noise_y = double(im_noise)/255;

if use_gpu
    im_noise_y = gpuArray(im_noise_y);
end


im_des_y = des_ds_Matconvnet(im_noise_y, model, v);


if use_gpu
    im_des_y = gather(im_des_y);
    im_noise_y = gather(im_noise_y);
end
figure,imshow(im_noise_y);
figure,imshow(im_des_y);



end
