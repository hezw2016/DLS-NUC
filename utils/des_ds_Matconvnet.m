function im_des_y = des_ds_Matconvnet(im_noise_y,model,v)
[lh,lw] = size(im_noise_y);
weight = model.weight;
bias = model.bias;
% layer_num = size(weight,2);
pad = floor((size(weight{1},1)-1)/2);
im_noise_y = single(im_noise_y);

switch v
    case 1
        convfea = vl_nnconv(im_noise_y,weight{1},bias{1},'Pad',pad);
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{2},bias{2},'Pad',pad);
        convfea = vl_nnrelu(convfea); 
        convfea = vl_nnconv(convfea,weight{3},bias{3},'Pad',pad);
        convfea3 = convfea;      
        for i = 4:7
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea); 
        end
        convfea = vl_nnconv(convfea,weight{8},bias{8},'Pad',pad);
        convfea8 = convfea;

        convfea = vl_nnpool(convfea3,3,'Stride',3,'Pad',0);

        for i= 11:20
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea); 
        end

        convfea = vl_nnconv(convfea,weight{21},bias{21},'Pad',pad);
        convfea = vl_nnconvt(convfea,weight{22},bias{22},'Upsample',3,'Crop',0 );
        sum1 = convfea + convfea8;

        convfea = vl_nnconv(sum1,weight{9},bias{9},'Pad',pad );
        convfea = vl_nnrelu(convfea); 
        convfea = vl_nnconv(convfea,weight{10},bias{10},'Pad',pad );
        im_des_y = im_noise_y - convfea;
        
    case 2
        convfea = vl_nnconv(im_noise_y,weight{1},bias{1},'Pad',pad);
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{2},bias{2},'Pad',pad);
        convfea2 = convfea;
        for i = 3 : 9
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea10 = vl_nnconv(convfea,weight{10},bias{10},'Pad',pad);
        
        convfea = vl_nnpool(convfea2,3,'Stride',3,'Pad',0);
        for i = 11 : 17
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea = vl_nnconv(convfea,weight{18},bias{18},'Pad',pad);
        convfea19 = vl_nnconvt(convfea,weight{19},bias{19},'Upsample',3,'Crop',0 );
        
        convfea = cat(3,convfea10,convfea19);
        convfea = vl_nnconv(convfea,weight{20},bias{20},'Pad',pad );
        im_des_y = im_noise_y - convfea;
    case 3
        convfea = vl_nnconv(im_noise_y,weight{1},bias{1},'Pad',pad);
        convfea = vl_nnrelu(convfea);
        convfea = vl_nnconv(convfea,weight{2},bias{2},'Pad',pad);
        convfea2 = convfea;
        
        convfea = vl_nnpool(convfea2,4,'Stride',4,'Pad',0);
        for i = 3 : 9
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea10 = vl_nnconv(convfea,weight{10},bias{10},'Pad',pad);
        convfea11 = vl_nnconvt(convfea10,weight{11},bias{11},'Upsample',4,'Crop',0 );
        
        convfea = vl_nnpool(convfea2,2,'Stride',2,'Pad',0);
        for i = 12 : 18
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea19 = vl_nnconv(convfea,weight{19},bias{19},'Pad',pad);
        convfea20 = vl_nnconvt(convfea19,weight{20},bias{20},'Upsample',2,'Crop',0 );
        
        convfea = cat(3,convfea2,convfea11,convfea20);
        convfea = vl_nnconv(convfea,weight{21},bias{21},'Pad',pad );
        im_des_y = im_noise_y - convfea;
    case 4
        convfea = vl_nnconv(im_noise_y,weight{1},bias{1},'Pad',pad);
        convfea1 = convfea;
        convfea = vl_nnpool(convfea1,3,'Stride',3,'Pad',0);
        for i = 2 : 8
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea9 = vl_nnconv(convfea,weight{9},bias{9},'Pad',pad);
        convfea10 = vl_nnconvt(convfea9,weight{10},bias{10},'Upsample',3,'Crop',0 );
        convfea = cat(3,convfea1,convfea10);
        convfea = vl_nnconv(convfea,weight{11},bias{11},'Pad',pad );
%         figure,imshow(convfea,[]);
        im_des_y = im_noise_y - convfea;
    case 5
        convfea = vl_nnconv(im_noise_y,weight{1},bias{1},'Pad',pad);
        convfea1 = convfea;
        convfea = vl_nnpool(convfea1,3,'Stride',3,'Pad',0);
        for i = 2 : 8
            convfea = vl_nnconv(convfea,weight{i},bias{i},'Pad',pad);
            convfea = vl_nnrelu(convfea);
        end
        convfea9 = vl_nnconv(convfea,weight{9},bias{9},'Pad',pad);
        convfea10 = vl_nnconvt(convfea9,weight{10},bias{10},'Upsample',3,'Crop',0 );
        convfea = cat(3,convfea1,convfea10);
        convfea = vl_nnconv(convfea,weight{11},bias{11},'Pad',pad );
%         figure,imshow(convfea,[]);
        im_des_y = convfea;
end