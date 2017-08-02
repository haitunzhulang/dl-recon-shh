function showcase_RMSE_vs_iteration_PLSTV()
% SHOWCASE_RMSE_VS_ITERATION_PLSTV ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 30-May-2017 14:47:36 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : showcase_RMSE_vs_iteration_PLSTV.m 
theta = (-25:24)*pi/180;
nrays = 256;

tic;[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);toc;

%%
NX = 256; NY = 256;
num_ellipses = 6;
[img,g] = simulate_projection(NX, theta,num_ellipses);
% g = g(:);
noise = .00;
g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
data.H = H;
data.g = g_noise(:);
data.true_img = img;

STEP_SIZE=.1;
TV_param = 0;%.0001;
VERBOSE=1;
cost_cutoff=.001;
MAX_ITER= 800;


[mse_error,df_loss,norm_grad,x] = fistav2d_bk(@cost_func_xray_H, zeros(NX,NX), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'max_iter',MAX_ITER,'proj_op','nonneg','min_norm_grad', cost_cutoff);


        
%%
clf;
subplot(1,4,1);
yyaxis left
semilogy(sqrt(mse_error(3:end)));
ylabel('RMSE_error');
yyaxis right
semilogy(df_loss(3:end));
ylabel('DF loss');

c_max = max([max(img(:)) max(x(:))]);
c_min = min([min(img(:)) min(x(:))]);

subplot(1,4,2);
imagesc(img);
caxis([c_min c_max]);
colorbar;
title('Original Image');
subplot(1,4,3);
imagesc(img-x);
colorbar;
title(['Difference image, RMSE: ' num2str(sqrt(mean(mean((img-x).^2))))]);
subplot(1,4,4);
imagesc(x);
caxis([c_min c_max]);
colorbar;
title('Reconstructed image');

%% Showcasing that these TV params are good TV params to select
STEP_SIZE=.75;
TV_params = [.0001 .0005 .001 .005];
% VERBOSE=2;
VERBOSE=1;
cost_cutoff=.01;
MAX_ITER= 800;
NX = 256; NY = 256;
x = zeros(256,256,1);
for tv=1:length(TV_params)
    [mse_error,df_loss,norm_grad,x(:,:,tv)] = fistav2d_bk(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, TV_params(tv), 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'max_iter',MAX_ITER,'proj_op','nonneg','min_rel_cost_diff', cost_cutoff);
end;

%
c_max = max([max(img(:)) max(x(:))]);
c_min = min([min(img(:)) min(x(:))]);
dif = x - repmat(img,[1 1 4]);
dif_max = max([max(dif(:))]);
dif_min = min([min(dif(:))]);

clf;
% set(gcf,'position',[234 413 1276 505]);
set(gcf,'color','w');
[ha, pos] = tight_subplot(2,5,[.05 .01],[.01 .1],[.01 .01]);
axes(ha(1));
imagesc(img);
caxis([c_min c_max]);
axis off
title('Original Image');


for i=1:length(TV_params)
    axes(ha(1+i));
    imagesc(squeeze(x(:,:,i)));
    caxis([c_min c_max]);
    title(['TV Weight: ' num2str(TV_params(i))]);
    axis off
    
    axes(ha(6+i));
    imagesc(squeeze(dif(:,:,i)));
    caxis([dif_min dif_max]);
    title(['RMSE: ' num2str(sqrt(mean(mean(dif(:,:,i)).^2)))]);
    axis off
end;
axes(ha(6));
hold on;
y_vert = 256/2;
plot(img(y_vert,:),'k');
plot(x(y_vert,:,1),'b');
plot(x(y_vert,:,2),'r');
plot(x(y_vert,:,3),'g');
plot(x(y_vert,:,4),'c');
legend('Original Image',['TV:' num2str(TV_params(1))],['TV:' num2str(TV_params(2))],...
    ['TV:' num2str(TV_params(3))],['TV:' num2str(TV_params(4))]);
title('Profile Plot');
axis off;

       

%% Jun 6 -- changing cost cutoff


NX = 256; NY = 256;
num_ellipses = 6;
[img,g] = simulate_projection(NX, theta,num_ellipses);
% g = g(:);
noise = .00;
g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
data.H = H;
data.g = g_noise(:);
data.true_img = img;

STEP_SIZE=.1;
TV_param = .001;%.0001;
VERBOSE=1;
cost_cutoff=.01;
MAX_ITER= 800;


[mse_error,df_loss,norm_grad,x] = fistav2d_bk(@cost_func_xray_H, zeros(NX,NX), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'max_iter',MAX_ITER,'proj_op','nonneg','min_norm_grad', cost_cutoff);

% Show it off
clf;
subplot(1,4,1);
yyaxis left
semilogy(sqrt(mse_error(3:end)));
ylabel('RMSE_error');
yyaxis right
semilogy(df_loss(3:end));
ylabel('DF loss');

c_max = max([max(img(:)) max(x(:))]);
c_min = min([min(img(:)) min(x(:))]);

subplot(1,4,2);
imagesc(img);
caxis([c_min c_max]);
colorbar;
title('Original Image');
subplot(1,4,3);
imagesc(img-x);
colorbar;
title(['Difference image, RMSE: ' num2str(sqrt(mean(mean((img-x).^2))))]);
subplot(1,4,4);
imagesc(x);
caxis([c_min c_max]);
colorbar;
title('Reconstructed image');

%% Comparing how the intialization affects result of PLS-TV
NX = 256; NY = 256;
num_ellipses = 6;
[img,g] = simulate_projection(NX, theta,num_ellipses);
% g = g(:);
noise = .00;
g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
data.H = H;
data.g = g_noise(:);
data.true_img = img;

STEP_SIZE=.1;
TV_param = .001;%.0001;
VERBOSE=1;
cost_cutoff=.01;
MAX_ITER= 800;

mse_error = {}; df_loss = {}; norm_grad = {}; x = {};
rand_start = {};
for i=1:5
    rand_start{i} = rand(NX,NX);
    if i==5
        rand_start{i}= img;
    end;
    [mse_error{i},df_loss{i},norm_grad{i},x{i}] = fistav2d_bk(@cost_func_xray_H, zeros(NX,NX), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'max_iter',MAX_ITER,'proj_op','nonneg','min_norm_grad', cost_cutoff);
    
end;

%
clf;
close all; clf; set(gcf,'position',[680 70 1715 889]);
set(gcf,'color','w');
[ha, pos] = tight_subplot(4,6,[.03 .05],[.02 .03],[.01 .03]);

axes(ha(1));
imagesc(img);
title('Original Image');
axis off;
for i=1:5
    axes(ha(i+1));
    imagesc(x{i});
    axis off;
    if i==1
        title('Reconstruction');
    end;
    
    axes(ha(i+7));
    imagesc(rand_start{i});
    axis off;
    if i==1
        title('Intialization of f');
    end;
    
    axes(ha(i+13));
    imagesc(img-x{i});
    axis off;
    if i==1
        title('Difference Image');
    end;
    
    axes(ha(i+19));
    yyaxis left
    semilogy(sqrt(mse_error{i}(3:end)));
    if i==1
        ylabel('RMSE error');
    end;
    yyaxis right
    semilogy(df_loss{i}(3:end));
    if i==1
        ylabel('DF loss');
        title('Loss during PLS-TV');
    end;
end;
axes(ha(7)); axis off; axes(ha(13)); axis off; axes(ha(19)); axis off;












 
% ===== EOF ====== [showcase_RMSE_vs_iteration_PLSTV.m] ======  
