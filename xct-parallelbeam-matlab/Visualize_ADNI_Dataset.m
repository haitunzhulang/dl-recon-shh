function Visualize_ADNI_Dataset()
% VISUALIZE_ADNI_DATASET ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 31-May-2017 12:22:51 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : Visualize_ADNI_Dataset.m 
addpath NIfTI_20140122

%%

mf = '../medception/ADNI/ADNI/';
files = dir([mf '*/*/*/*/*.nii']);


%%
for i=1:2
    %%
    fname = [files(i).folder '/' files(i).name];
    data = load_nii(fname);
    img = data.img;
    
    %%
    sz = size(img);
    for j=1:sz(1)
        imagesc(squeeze(img(j,:,:)));
        title(['Slice ' num2str(j)]);
        pause(.05);
    end;
    
    %%
    
    sz = size(img);
    for j=1:sz(2)
        imagesc(squeeze(img(:,j,:)));
        title(['Slice ' num2str(j)]);
        pause(.05);
    end;
    
    
    %% This is the one we want!!!
    sz = size(img);
    for j=200:sz(3)
        imagesc(squeeze(img(:,:,j)));
        title(['Slice ' num2str(j)]);
        pause;
    end;
    
end;



%% Plot the number of pixels which are non 0

subplot(1,2,1);
aa = img ~=0;
non_0_count = squeeze(sum(squeeze(sum(aa,1)),1));
plot(non_0_count);
ylabel('Number of pixels which are > 0');
xlabel('Different slices');
subplot(1,2,2);
imagesc(squeeze(img(:,:,220)));
title('Slice 220');




%% Load up images
num_images = 50;
start_index=21;
data = {};
for i=1:num_images
    fname = [files(i+start_index).folder '/' files(i+start_index).name];
    data{i} = load_nii(fname);
end;  

%% Are the scans of the same dimensionality?
% Compare shapes -- mostly the same!
shapes = zeros(1,3);
for i=1:num_images
    shapes(i,:) = size(data{i}.img);
end; 

figure(2);
plot(shapes);

%% Showing a bunch at once
clf;
set(gcf,'color','w');
[ha,pos] = tight_subplot(5,10,[.01 .01],[.01 .01],[.01 .01]);

% Show images
for j=1:256
    for i=1:num_images
        axes(ha(i));
        imagesc(squeeze(data{i}.img(:,:,j)));
        axis off;
    end;
    drawnow();
    pause(.01);
end;
    
%% Standardize to be between 0 and 1, then decide on cutoff

z = 90;
img = squeeze(data{4}.img(:,:,:));
img = rescale_array(img,0,1);

count_0 = sum(img(:)==0);

cutoff = .1;
mask = img<cutoff;

clf;
subplot(2,2,1);
imagesc(squeeze(img));
% colorbar;
subplot(2,2,2);
imagesc(mask);
title(['Mask with cutoff: ' num2str(cutoff)]);


CC = bwconncomp(mask);

numPixels = cellfun(@numel,CC.PixelIdxList);
[biggest,idx] = max(numPixels);

masked_img = img;
masked_img(CC.PixelIdxList{idx}) = 0;
subplot(2,2,3);
imagesc(masked_img);
title('Masked Image');
% smooth out the masked image:
% h = fspecial('gaussian', [3 3], .25);
% smoothed_masked_img = imfilter(masked_img, h, 'replicate');

% subplot(2,2,4);
% imagesc(smoothed_masked_img);
% title('Smoothed masked image');


count_0_after = sum(masked_img(:)==0);
sz = size(img); total = sz(1)*sz(2);

disp(['Count of 0 pixels before: ' num2str(count_0) ', after: ' num2str(count_0_after)]);
disp(['This is a change from ' num2str(count_0/total) ' to ' num2str(count_0_after/total)]);

%% Use magnitude of gradient to select cutoff -- need to generate mask
% 
% img = squeeze(data{1}.img(:,:,120));
% 
% [Gx Gy] = imgradientxy(img);    
% subplot(2,3,1);
% imagesc(img);    
% subplot(2,3,2);
% imagesc(Gx);
% title('Gradient in x direction');
% subplot(2,3,3);
% imagesc(Gy);
% title('Gradient in y direction');
% 
% 
% subplot(2,3,5);
% hist(Gx(:));
% subplot(2,3,6);
% hist(Gy(:));
%   

%% Bluring images
img = squeeze(data{20}.img(:,:,120));
std = 1;
b_img = imgaussfilt(img,std);

[Gx Gy] = imgradientxy(img);  
[b_Gx b_Gy] = imgradientxy(b_img);  

clf;
subplot(4,3,1);
imagesc(img);


subplot(4,3,2);
hist(Gx(:));
title('Hist of G in x direction');
subplot(4,3,3);
hist(Gy(:));
title('Hist of G in y direction');

subplot(4,3,5);
imagesc(Gx);
title('Gradient in x direction');
subplot(4,3,6);
imagesc(Gx);
title('Gradient in y direction');

subplot(4,3,7);
imagesc(b_img);
title(['Blurred by ' num2str(std)]);


subplot(4,3,8);
hist(b_Gx(:));
title('Hist of G in x direction');
subplot(4,3,9);
hist(b_Gy(:));
title('Hist of G in y direction');

subplot(4,3,11);
imagesc(b_Gx);
title('Gradient in x direction');
subplot(4,3,12);
imagesc(b_Gx);
title('Gradient in y direction');
 
%% Step 1, Remove noise around brain
img = squeeze(data{6}.img);
masked_img = Standardize_and_apply_mask(img);

sz = size(img);

sz = size(masked_img);
if (sum(squeeze(sum(masked_img(sz(1),:,:)))) > 0)
    disp('Yes, right up to boundary');
else
    disp('No, not right up to boundary');
end;
%%
for i=150:sz(3)
    subplot(1,2,1);
    imagesc(squeeze(img(:,:,i)));
    title(['Original image.  Slice ' num2str(i)]);
    
    subplot(1,2,2);
    imagesc(squeeze(masked_img(:,:,i)));
    title('Masked');
    
    
    drawnow();
    pause();
    
end;

%% Is image right up to boundary?  If yes, trash
sz = size(masked_img);
if sum(masked_img(sz(1)-6,floor(sz(2/2))) > 0
    disp('Yes, right up to boundary');
else
    disp('No, not right up to boundary');
end;


%% Is image mostly empty?
% img = squeeze(data{6}.img);
img = squeeze(data{7}.img);
masked_img = Standardize_and_apply_mask(img);

non_0_pixels = squeeze(sum(sum(masked_img ~=0,1),2));
sz = size(masked_img);
total_pixels = sz(1)*sz(2);


subplot(1,2,1);
z = 225;
imagesc(squeeze(masked_img(:,:,z)));
title(['Slice at ' num2str(z)]);
subplot(1,2,2);
plot(non_0_pixels/total_pixels);
ylabel('Percentage of non 0 pixels');
xlabel('Different slices');

%% Emptiness cutoff tests
img = squeeze(data{7}.img);
masked_img = Standardize_and_apply_mask(img);

non_0_pixels = squeeze(sum(sum(masked_img ~=0,1),2));
sz = size(masked_img);
total_pixels = sz(1)*sz(2);


non_0_pixels = non_0_pixels/total_pixels;
cutoff = .25;

remove_indices = find(non_0_pixels <.25);

subplot(1,3,1);
ind = 3;
imagesc(squeeze(masked_img(:,:,remove_indices(ind)-1)));
title(['Image just before cutoff: ' num2str(remove_indices(ind)-1) ...
    ', with percentage: ' num2str(non_0_pixels(remove_indices(ind)-1))]);

subplot(1,3,2);
ind = 3;
imagesc(squeeze(masked_img(:,:,remove_indices(ind))));
title(['Image just at cutoff: ' num2str(remove_indices(ind)) ...
    ', with percentage: ' num2str(non_0_pixels(remove_indices(ind)))]);

subplot(1,3,3);
ind = 4;
imagesc(squeeze(masked_img(:,:,remove_indices(ind))));
title(['Image just after cutoff: ' num2str(remove_indices(ind)) ...
    ', with percentage: ' num2str(non_0_pixels(remove_indices(ind)))]);


%% Putting it all together

mf = '../medception/ADNI/ADNI/';
files = dir([mf '*/*/*/*/*.nii']);

% For mostly empty test
cutoff = .25;
%% Load up images, save them somewhere
save_location = 'medical_images/';
final_image_size = [300 300];
clf;
count = 0;
writing_images = 0;
for i=1:length(files)
    %%
    fname = [files(i).folder '/' files(i).name];
    data = load_nii(fname);
    img = data.img;
    masked_img = Standardize_and_apply_mask(img);
    
    % If image goes right up to boundary, skip, else do it
    sz = size(masked_img);
    if sum(abs(masked_img(sz(1)-6,floor(sz(2/2))))) == 0
        
        non_0_pixels = squeeze(sum(sum(masked_img ~=0,1),2));
        sz = size(masked_img);
        total_pixels = sz(1)*sz(2);
        non_0_pixels = non_0_pixels/total_pixels;
        for j=10:sz(3)
            % Is image mostly empty? if no then save it!
            if non_0_pixels(j) > cutoff
                subplot(1,3,1);
                imagesc(squeeze(img(:,:,j)));
                title('Original Image');
                subplot(1,3,2);
                imagesc(squeeze(masked_img(:,:,j)));
                title(['Slice ' num2str(j) ' from ' num2str(i)]);
                [NA, y_est] = BM3D(1, squeeze(masked_img(:,:,j)), 5); 
                subplot(1,3,3);
                imagesc(y_est);
                title(['Smoothed with BM3D, ' num2str(count)]);
                drawnow();
                pause(.01);
                
                final_img = zeros(final_image_size);
                dif_1 = floor((final_image_size(1)-sz(1))/2);
                dif_2 = floor((final_image_size(2)-sz(2))/2);
                final_img(dif_1:dif_1+sz(1)-1,dif_2:dif_2+sz(2)-1) = y_est;
                fname = [save_location files(i).name '_' num2str(j) '.png'];
                if writing_images
                    imwrite(final_img,fname);
                end;
                count = count +1;
            else
                imagesc(squeeze(masked_img(:,:,j)));
                title(['EMPTY. Slice ' num2str(j) ' from ' num2str(i)]);
                drawnow();
                pause(.01);
            end;
        end;
    else
        imagesc(squeeze(masked_img(:,:,60)));
        title(['Skipping this one (' num2str(i) ') because it goes up to the edge!']);
        drawnow();
        pause();
    end;
end;  


%% Smoothing images - creating example images to show to blog
addpath BM3D

save_location = 'medical_images/';
final_image_size = [300 300];
clf;
count = 0;
writing_images = 0;
i=1;
fname = [files(i).folder '/' files(i).name];
data = load_nii(fname);
img = data.img;
masked_img = Standardize_and_apply_mask(img);
non_0_pixels = squeeze(sum(sum(masked_img ~=0,1),2));
sz = size(masked_img);
total_pixels = sz(1)*sz(2);
non_0_pixels = non_0_pixels/total_pixels;
%%
% addpath BM3D
clf;
set(gcf,'color','w');
set(gcf,'position',[256 518 923 356]);
[ha, pos] = tight_subplot(1,3,[.01 .01],[.03 .1],[.01 .01]);
j=150;
axes(ha(1));
imagesc(squeeze(img(:,:,j)));
title('Original Image');
axis off;
axes(ha(2));
imagesc(squeeze(masked_img(:,:,j)));
title(['Slice ' num2str(j) ' from ' num2str(i)]);
axis off;
[NA, y_est] = BM3D(1, squeeze(masked_img(:,:,j)), 5); 
axes(ha(3));
imagesc(y_est);
title(['Smoothed with BM3D']);
axis off;




    
% ===== EOF ====== [Visualize_ADNI_Dataset.m] ======  
