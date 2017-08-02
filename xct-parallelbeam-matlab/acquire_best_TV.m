function [img, best_mse] = acquire_best_TV(g,f_true,H,support,index,tv_storage)
% ACQUIRE_BEST_TV ... 
%  

% This function will line search to find the best tv param for this image
% according to it's immse score, and return that image/immse score
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 05-Apr-2017 17:25:47 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : acquire_best_TV.m 
%%
% keyboard();
% Try to load if exists
fname = [tv_storage num2str(index) '.mat'];
previous_calculation = dir(fname);
VERBOSE=0;
DEBUG=1;

if length(previous_calculation)==1
    load([fname]);
    [Y,I] = min(mses);
    best_mse = Y;
    img = squeeze(img_recons(I,:,:));
    if VERBOSE
        [Y,I] = sort(reg_params,'descend');
        for i=1:length(reg_params)
            subplot(4,4,i);
            imagesc(squeeze(img_recons((i),:,:)));
            title(['W: ' num2str(reg_params(I(i))) '. MSE: ' num2str(mses(I(i)))]);
        end;
        subplot(4,4,16);
        semilogx(reg_params(I),mses(I),'^-');
        title('Mean squared errors vs Weight Parameter');
        xlabel('Weight Parameter');
        ylabel('MSE')
        axis([-inf inf min(mses(I(2:end)))*.9 max(mses(I(2:end)))*1.1]);
        drawnow();
        pause();
    end;
    return;
end;

drawing = 0;
% reg_params = [0.005,.0025,.001,.00075,.0005,.00025,.0001,.000075,.00005,.000025,.00001,.000005,.0000005,.00000005,0];
cost_cutoff=.001;

data.H = H;
data.g = double(g(:));

STEP_SIZE=.75;
MAX_ITER=300;
NX=256;
NY=256;
% img_recons = zeros([length(reg_params) 256 256]);
VERBOSE=0;

if drawing
    clf;
    subplot(4,4,1);
    imagesc(f_true);
    title('True phantom');
    drawnow();
    pause(.01);
end;

%% implement binary serach

right_index=100;
param_grid = logspace(-2,-11,right_index);
left_index = 1;
middle_index=floor((right_index+left_index)/2);

reg_params = [];
reg_params(1) = param_grid(left_index);
reg_params(2) = param_grid(middle_index);
reg_params(3) = param_grid(right_index);
mses = zeros(1,10);
img_recons = zeros([10,256,256]);
count=1;
for i=1:3
    img_recons(i,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, reg_params(i), ...
        'output_filename_prefix', '', 'verbose', VERBOSE, ...
        'min_rel_cost_diff', cost_cutoff,...
        'max_iter', MAX_ITER, 'proj_op','nonneg');
    mses(i) = immse(squeeze(img_recons(i,support(1):support(2),support(3):support(4))),f_true(support(1):support(2),support(3):support(4)));
    
    if drawing
        subplot(4,4,count+1);
        imagesc(squeeze(img_recons(i,:,:)));
        title(['Reg: ' num2str(log10(reg_params(i))) '. psnr: ' num2str((mses(i)))]);
        drawnow();
        pause(.01);
    end;
    count = count+1;
end;

% count=4;
can_still_expand=1;
mse_center = mses(1);
while can_still_expand
    disp(['on count: ' num2str(count) ' for index: ' num2str(index)]);
    expand_left_index = floor((middle_index+left_index)/2);
    expand_right_index=floor((middle_index+right_index)/2);
    
    reg_params(count) = param_grid(expand_left_index);
    img_recons(count,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, STEP_SIZE, reg_params(count),'output_filename_prefix', '', 'verbose', VERBOSE, 'min_rel_cost_diff', cost_cutoff,'max_iter', MAX_ITER, 'proj_op','nonneg');
    mses(count) = immse(squeeze(img_recons(count,support(1):support(2),support(3):support(4))),f_true(support(1):support(2),support(3):support(4)));
    mse_left = mses(count);
    if drawing
        subplot(4,4,count+1);
        imagesc(squeeze(img_recons(count,:,:)));
        title(['Reg: ' num2str(log10(reg_params(count))) '. psnr: ' num2str((mses(count)))]);
    end;
    count = count+1;
    
    reg_params(count) = param_grid(expand_right_index);
    img_recons(count,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, STEP_SIZE, reg_params(count),'output_filename_prefix', '', 'verbose', VERBOSE, 'min_rel_cost_diff', cost_cutoff,'max_iter', MAX_ITER, 'proj_op','nonneg');
    mses(count) = immse(squeeze(img_recons(count,support(1):support(2),support(3):support(4))),f_true(support(1):support(2),support(3):support(4)));
    mse_right = mses(count);
    if drawing
        subplot(4,4,count+1);
        imagesc(squeeze(img_recons(count,:,:)));
        title(['Reg: ' num2str(log10(reg_params(count))) '. psnr: ' num2str((mses(count)))]);
    end;
    count = count+1;
    if DEBUG; disp(['MSE center: ' num2str(mse_center) '. Left: ' num2str(mse_left) '. Right: ' num2str(mse_right)]);end;
    % if center has lowest MSE, then expand inside left and right
    if mse_center < mse_left && mse_center < mse_right
        left_index = expand_left_index;
        right_index = expand_right_index;
        if DEBUG; disp('Taking step Center!');end;
    else
        % if left index has lower mse than right index, expand that direction.
        % else expand right
        if mse_left < mse_right
            tmp_right = middle_index;
            middle_index=expand_left_index;
            right_index=tmp_right;
            mse_center = mse_left;
            if DEBUG; disp('Taking step Left!');end;
        else
            tmp_left = middle_index;
            middle_index=expand_right_index;
            left_index=tmp_left;
            mse_center = mse_right;
            if DEBUG; disp('Taking step Right!');end;
        end;
    end;
    % Keep expanding if there is still room
    if abs(left_index-right_index) < 6
        can_still_expand=0;
    end;
    
    drawnow();
    pause(.01);
end;

% for i=1:length(reg_params)
%     disp(['solving tv prob for index:' num2str(index) ' on param : ' num2str(i)]);
%     reg_param = reg_params(i);
%     img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
%             STEP_SIZE, reg_param, ...
%         'output_filename_prefix', '', 'verbose', VERBOSE, ...
%         'min_rel_cost_diff', cost_cutoff,...
%         'max_iter', MAX_ITER, 'proj_op','nonneg');
%     img_recons(i,:,:) = img_recon;
%     
% end;

% Calc mse scores and return them!
%%
% best_mse = 1000000000;
% best_ind = 0;
% 
% mses = zeros(1,length(reg_params));
% 
% for i=1:length(reg_params)
%     
%     mse_i = immse(squeeze(img_recons(i,support(1):support(2),support(3):support(4))),f_true(support(1):support(2),support(3):support(4)));
%     mses(i) = mse_i;
%     if drawing
%         subplot(4,4,i+1);
%         imagesc(squeeze(img_recons(i,:,:)));
%         title(['Reg: ' num2str(reg_params(i)) '. mse : ' num2str(mse_i) '. psnr: ' num2str(convert_mse_to_psnr(mse_i))]);
%     end;
%    
%     if mse_i < best_mse
%         best_mse = mse_i;
%         best_ind = i;
%     end;
% end;
% if drawing
%     subplot(4,4,best_ind+1);
%     title(['BEST  Reg: ' num2str(reg_params(best_ind)) ' mse : ' num2str(best_mse) '. psnr: ' num2str(convert_mse_to_psnr(best_mse))]);
%     drawnow();
%     %     pause();
% end;

% if mses(1)<mses(2) || mses(end-1) < mses(end-2)
%     disp(['Aww crap, the grid wasnt big enough for this particular example : ' num2str(mses)]);
% %     pause();
%     
% end;

[Y,I] = min(mses);
best_mse = Y;
img = squeeze(img_recons(I,:,:));

save(fname,'img', 'best_mse','img_recons','mses','reg_params');


 
% ===== EOF ====== [acquire_best_TV.m] ======  
