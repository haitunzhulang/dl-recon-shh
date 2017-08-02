function masked_img = Standardize_and_apply_mask(img)
% STANDARDIZE_AND_APPLY_MASK ... 
%  
%  img is a 3d img

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 01-Jun-2017 12:57:38 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : Standardize_and_apply_mask.m 

drawing = 0;
img = rescale_array(img,0,1);
% img = padarray(img,[6 6 0]);

masked_img = img;

% SMOOTH_FILTER_SIZE = [9, 9];
% SMOOTH_FILTER_WIDTH = 1;
% h = fspecial('gaussian', SMOOTH_FILTER_SIZE, SMOOTH_FILTER_WIDTH);
% img = imfilter(img, h, 'replicate');

cutoff = .1;
mask = double(img<cutoff);




sz = size(img);
for j=1:sz(3)
    
    SMOOTH_FILTER_SIZE = [5 5];
    SMOOTH_FILTER_WIDTH = 1;
    h = fspecial('gaussian', SMOOTH_FILTER_SIZE, SMOOTH_FILTER_WIDTH);
    mask(:,:,j) = imfilter(squeeze(mask(:,:,j)), h, 'replicate');
    mask(:,:,j) = mask(:,:,j) == 1;
    
      
    CC = bwconncomp(squeeze(mask(:,:,j)));
    
    
    numPixels = cellfun(@numel,CC.PixelIdxList);
    [biggest,idx] = max(numPixels);
    
    tmp_img = squeeze(masked_img(:,:,j));
    tmp_img(CC.PixelIdxList{idx}) = 0;
    masked_img(:,:,j) = tmp_img;
end;

if drawing
    
    z = 90;
    
    clf;
    subplot(2,2,1);
    imagesc(squeeze(img(:,:,z)));
    % colorbar;
    subplot(2,2,2);
    imagesc(squeeze(mask(:,:,z)));
    title(['Mask with cutoff: ' num2str(cutoff) ' at slice ' num2str(z)]);
    
    subplot(2,2,3);
    imagesc(squeeze(masked_img(:,:,z)));
    title('Masked Image');
end;

% smooth out the masked image:
% h = fspecial('gaussian', [3 3], .25);
% smoothed_masked_img = imfilter(masked_img, h, 'replicate');

% subplot(2,2,4);
% imagesc(smoothed_masked_img);
% title('Smoothed masked image');


% count_0_after = sum(masked_img(:)==0);
% sz = size(img); total = sz(1)*sz(2);





 
% ===== EOF ====== [Standardize_and_apply_mask.m] ======  
