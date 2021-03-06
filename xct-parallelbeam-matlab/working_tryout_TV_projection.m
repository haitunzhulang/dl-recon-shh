function working_tryout_TV_projection()
% WORKING_TRYOUT_TV_PROJECTION ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 13-Jun-2017 15:23:11 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : working_tryout_TV_projection.m 

addpath toolbox_image
addpath toolbox_image/toolbox
addpath ../fista-matlab

%% Load Image
% img = double(imread('coins.png'))/256;
% imagesc(img/256);
% colormap gray
theta = (-24:25)*pi/180;
nrays = 256;
[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);

NX = 256;
NY = 256;
[img,g,ellipses] = load_sim_projection(1,NX,theta);


%%
data.g = g(:);
data.H = H;


STEP_SIZE=.75;
TV_param = 0;
gamma = 1000;
cutoff = .01;
%%
img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', 2, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gamma);

%%

noisy_img = imnoise(img,'gaussian');
gammas = [100 500 1000];

options.niter = 50;
options.verbose = 0;
options.xtgt = img;
[tv_projection, err_tv,err_12,err_tgt] = perform_tv_projection(noisy_img,gammas(2),options);



% tv_projection_in_action
figure(1);
clf;
subplot(1,4,1);
imagesc(img);
title('Original Image');
colorbar;
subplot(1,4,2);
imagesc(noisy_img);
title('Image with some gaussian noise');
colorbar;
subplot(1,4,3);
imagesc(tv_projection);
title('Projection onto TV Ball');
colorbar;
subplot(1,4,4);
imagesc(noisy_img-tv_projection);
title('Dif NoisyImg - TVProjection');
colorbar;

%%
% How gamma affects the best iteration number
figure(2);
clf;
subplot(1,4,1);
hold on;
[tv_projection1, err_tv,err_12,err_tgt] = perform_tv_projection(noisy_img,gammas(1),options);
plot(err_tgt,'r');
[tv_projection2, err_tv,err_12,err_tgt] = perform_tv_projection(noisy_img,gammas(2),options);
plot(err_tgt,'b');
[tv_projection3, err_tv,err_12,err_tgt] = perform_tv_projection(noisy_img,gammas(3),options);
plot(err_tgt,'g');
xlabel('Iteration');
ylabel('MSE');
legend(['Gamma: ' num2str(gammas(1))],['Gamma: ' num2str(gammas(2))],['Gamma: ' num2str(gammas(3))]);
title('What iteration to select for a different Gammas');
subplot(1,4,2);
imagesc(tv_projection1);
title(['Gamma: ' num2str(gammas(1))]);
subplot(1,4,3);
imagesc(tv_projection2);
title(['Gamma: ' num2str(gammas(2))]);
subplot(1,4,4);
imagesc(tv_projection3);
title(['Gamma: ' num2str(gammas(3))]);




%% So what is the appropriate gamma? - 2nd Day Wed 6/14/17
theta = (-24:25)*pi/180;
nrays = 256;
[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);
NX = 256;
NY = 256;
[img,g,ellipses] = load_sim_projection(2,NX,theta);
data.g = g(:);
data.H = H;
noise = .02;
g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
data.g = g_noise(:);
%%
STEP_SIZE=.75;
TV_param = 0;
gamma = 1000;
cutoff = .01;
VERBOSE=2;
img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gamma);

err = mean((img_recon(:) - img(:)).^2);

%% How does different random initialization affect the reconstruction?
VERBOSE=1;
cutoff = .001;
gammas = [100 500 1000 inf];
num_rand_inits = 5;
rand_inits = rand([num_rand_inits+1 NX NX]);
rand_inits(num_rand_inits,:,:) = img;
rand_inits(4,:,150:end) = 0;
recons = zeros([length(gammas) num_rand_inits NX NX]);
for i=1:length(gammas)
    for j=1:num_rand_inits
        recons(i,j,:,:) = fistatv2d(@cost_func_xray_H, squeeze(rand_inits(j,:,:)), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gammas(i));
    end;
end;

%%
figure(1);
clf;
[ha,pos] = tight_subplot(5,5,[.01 .01],[.01 .03], [.01 .01]);
set(gcf,'Position',[280 61 1194 898]);
set(gcf,'color','w');
start = 0;
for i=1:num_rand_inits
    rand_init = rand(NX);
    axes(ha(start + 1));
    imagesc(squeeze(rand_inits(i,:,:)));
    axis off;
    if i==1
        title('Initialization');
    end;
    for j=1:length(gammas)
        axes(ha(start+1+j));
        
        imagesc(squeeze(recons(j,i,:,:)));
        axis off;
        if i==1
            title(['Recon with Gamma: ' num2str(gammas(j))]);
        end;
    end;
    start = start+5;
end;

%% How does Gamma affect the MSE of the recon?  What Gamma should we use?
[img,g,ellipses] = load_sim_projection(2,NX,theta);
TV = compute_total_variation(img);
disp(['TV of image is :'  num2str(TV)]);
gammas = [TV*.3 TV*.5 TV*.7 TV*.9 TV TV*1.1 TV*1.3 TV*1.5 TV*100];
data.g = g(:);
data.H = H;
noise = .02;
g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
data.g = g_noise(:);

% gammas = logspace(2,4,12);
recons = zeros([length(gammas) NX NX]);

for i=1:length(gammas)
    recons(i,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gammas(i));
end;

%% 
differences = zeros([1 length(gammas)]);
df_cost = zeros([1 length(gammas)]);
figure(2);
clf;
[ha,pos] = tight_subplot(5,5,[.03 .02],[.01 .03], [.01 .01]);


set(gcf,'Position',[280 61 1194 898]);
set(gcf,'color','w');
axes(ha(1));
imagesc(img);
axis off;
title(['Original Image, TV:' num2str(TV)]);
for i=1:length(gammas)
    axes(ha(1+i));
    aa = squeeze(recons(i,:,:));
    imagesc(aa);
    axis off;
    differences(i) = mean((aa(:)-img(:)).^2);
    title(['TVC:' num2str(round(gammas(i))) ', MSE:' num2str(round(differences(i)*100000))]);
    df_cost(i) = mean(data.g - data.H*aa(:));
    
end;

axes(ha(2));
figure(3);
clf;
% hold on;
semilogx(gammas,differences,'bo-');
% semilogx(gammas,df_cost,'ro-');
% axis off;
xlabel('TV Constraint');
ylabel('MSE');
title('MSE between Recon and Original - Inverse Crime 180 Degrees');

[Y,I] = min(differences);
disp(['Minimum MSE: ' num2str(Y) ', at TVC: ' num2str(gammas(I))]);

%%

noisy_img = imnoise(img,'gaussian');
gammas = [100 500 1000];

options.niter = 50;
options.verbose = 0;
options.xtgt = img;
[tv_projection, err_tv,err_12,err_tgt] = perform_tv_projection(noisy_img,gammas(2),options);

%% Inverse Crime with full field of view Data
theta = (-89:90)*pi/180;
nrays = 64;
tic;[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);toc;

NX = nrays;
[img,g,ellipses] = load_sim_projection(2,NX,theta);
g_inverse_crime = H*img(:);

data.H = H;
data.g = g_inverse_crime(:);

TV = compute_total_variation(img);
disp(['TV of image is :'  num2str(TV)]);
gammas = [TV*.3 TV*.5 TV*.7 TV*.9 TV TV*1.1 TV*1.3 TV*1.5 TV*100];
% gammas = logspace(2,4,12);
recons = zeros([length(gammas) NX NX]);

%
for i=1:length(gammas)
    recons(i,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
            .05, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gammas(i));
end;
differences = zeros([1 length(gammas)]);
df_cost = zeros([1 length(gammas)]);
figure(2);
clf;
[ha,pos] = tight_subplot(5,5,[.03 .02],[.01 .03], [.01 .01]);


set(gcf,'Position',[280 61 1194 898]);
set(gcf,'color','w');
axes(ha(1));
imagesc(img);
axis off;
title(['Original Image, TV:' num2str(TV)]);
for i=1:length(gammas)
    axes(ha(1+i));
    aa = squeeze(recons(i,:,:));
    imagesc(aa);
    axis off;
    differences(i) = mean((aa(:)-img(:)).^2);
    title(['TVC:' num2str(round(gammas(i))) ', MSE:' num2str(round(differences(i)*100000))]);
    df_cost(i) = mean(data.g - data.H*aa(:));
    
end;

axes(ha(2));
figure(3);
clf;
% hold on;
semilogx(gammas,differences,'bo-');
% semilogx(gammas,df_cost,'ro-');
% axis off;
xlabel('TV Constraint');
ylabel('MSE');
title('MSE between Recon and Original - Inverse Crime 180 Degrees');

[Y,I] = min(differences);
disp(['Minimum MSE: ' num2str(Y) ', at TVC: ' num2str(gammas(I))]);

%% NON Inverse Crime with full field of view Data
theta = (-89:90)*pi/180;
nrays = 64;
tic;[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);toc;

NX = nrays;
[img,g,ellipses] = load_sim_projection(2,NX,theta);
% g_inverse_crime = H*img(:);

data.H = H;
data.g = g(:);

TV = compute_total_variation(img);
disp(['TV of image is :'  num2str(TV)]);
gammas = [TV*.3 TV*.5 TV*.7 TV*.9 TV TV*1.1 TV*1.3 TV*1.5 TV*100];
% gammas = logspace(2,4,12);
recons = zeros([length(gammas) NX NX]);

%
for i=1:length(gammas)
    recons(i,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
            .05, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
            'tv_projection_gamma',gammas(i));
end;

differences = zeros([1 length(gammas)]);
df_cost = zeros([1 length(gammas)]);
figure(2);
clf;
[ha,pos] = tight_subplot(5,5,[.03 .02],[.01 .03], [.01 .01]);


set(gcf,'Position',[280 61 1194 898]);
set(gcf,'color','w');
axes(ha(1));
imagesc(img);
axis off;
title(['Original Image, TV:' num2str(TV)]);
for i=1:length(gammas)
    axes(ha(1+i));
    aa = squeeze(recons(i,:,:));
    imagesc(aa);
    axis off;
    differences(i) = mean((aa(:)-img(:)).^2);
    title(['TVC:' num2str(round(gammas(i))) ', MSE:' num2str(round(differences(i)*100000))]);
    df_cost(i) = mean(data.g - data.H*aa(:));
    
end;

axes(ha(2));
figure(3);
clf;
% hold on;
semilogx(gammas,differences,'bo-');
% semilogx(gammas,df_cost,'ro-');
% axis off;
xlabel('TV Constraint');
ylabel('MSE');
title('MSE between Recon and Original - Non Inverse Crime 180 Degrees');

[Y,I] = min(differences);
disp(['Minimum MSE: ' num2str(Y) ', at TVC: ' num2str(gammas(I))]);
 
%% What is the correct TV param?
TV_values = zeros(1);
Range = 10;theta = (-1*(Range/2)+1:Range)*pi/180;
tic;
for i=1:7500
    [img,g,ellipses] = load_sim_projection(i,256,theta);
    TV_values(i) = compute_total_variation(img);
    if (mod(i,1000) == 0) toc;disp(['Done at i: ' num2str(i)]);tic;end;
end;

%

figure(1);clf;subplot(1,3,1);
hist(TV_values);
xlabel('TV of image');
ylabel('Number of images out of 7500');
title('Histogram of total variation of images in our dataset');
disp(['Mean:' num2str(mean(TV_values)) ', Max: ' num2str(max(TV_values)) ...
    ', Min: ' num2str(min(TV_values))]);
disp(['Std: ' num2str(std(TV_values)) ', 3std+mean: ' num2str(std(TV_values)*3+mean(TV_values))]);
subplot(1,3,2);
[Y,I] = min(TV_values);
[img,g,ellipses] = load_sim_projection(I,256,theta);
imagesc(img);
title('Image with least TV');
subplot(1,3,3);
[Y,I] = max(TV_values);
[img,g,ellipses] = load_sim_projection(I,256,theta);
imagesc(img);
title('Image with most TV');



%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Jul 3 - Finding Best FoV Range for Dataset
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for i=2:4
    Range = 50;
    num_tests = 8;
    figure(i);clf; [ha,pos] = tight_subplot(5,num_tests,[.03 .01],[.01 .03], [.01 .01]);
    set(gcf,'Position',[2 36 1918 923]);
    count = 1;
    STEP_SIZE=.5;cutoff = .01;VERBOSE=1;noise =.02;
    TV_Constraint = 838.1038;
    nrays = 256;NX=nrays; profile_ray = 256/2;
    for Range=Range:10:Range+(num_tests-1)*10

        theta = (-1*(Range/2)+1:Range)*pi/180;
        tic;[~, H] = calc_projs(ones(nrays,nrays), theta, nrays);disp('Calcing H');toc;
        [img,g,ellipses] = load_sim_projection(i,nrays,theta);
        g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);

        data.g = g(:);
        data.H = H;

        img_recon_normal_no_noise = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
                    STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
                    'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
                    'tv_projection_gamma',inf);

        data.g = g_noise(:);
        img_recon_normal_noise = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
                    STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
                    'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
                    'tv_projection_gamma',inf);

        data.g = g(:);
        img_recon_tv_constraint_no_noise = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
                    STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
                    'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
                    'tv_projection_gamma',TV_Constraint);

        data.g = g_noise(:);
        img_recon_tv_constraint_noise = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
                    STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
                    'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
                    'tv_projection_gamma',TV_Constraint);

        img_recon_PLSTV_noise = fistatv2d(@cost_func_xray_H, zeros(NX,NX), data, ...
                    STEP_SIZE, .001, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
                    'min_rel_cost_diff', cutoff,'max_iter',1000,'proj_op','nonneg', ...
                    'tv_projection_gamma',inf);

        axes(ha(count));
        imagesc(img_recon_normal_no_noise);
        title([num2str(Range) ' D, No TV, No Noise']);
        axis off;

    %     axes(ha(count+num_tests*2));
    %     imagesc(img_recon_normal_noise);
    %     title('No TV, Noise');
    %     axis off;

        axes(ha(count+num_tests*1));
        imagesc(img_recon_tv_constraint_no_noise);
        title('TVC, No Noise');
        axis off;

        axes(ha(count+num_tests*2));
        imagesc(img_recon_tv_constraint_noise);
        title('TVC, Noise');
        axis off;


        axes(ha(count+num_tests*3));
        imagesc(img_recon_PLSTV_noise);
        title('PLS-TV, Noise');
        axis off;

        axes(ha(count+num_tests*4));
        hold on;
        plot(squeeze(img(profile_ray,:)),'g');
        plot(squeeze(img_recon_tv_constraint_noise(profile_ray,:)),'b');
        plot(squeeze(img_recon_PLSTV_noise(profile_ray,:)),'r');
        if (Range==50)legend('Original Image','TVC, No Noise', 'PLS-TV, Noise');end;
        title('Profile Image');
        axis off;

        count=count+1;
        drawnow();
        pause(.01);
    end;    
end  
% figure(3);
% imagesc(img);
% title('Target Image');























% ===== EOF ====== [working_tryout_TV_projection.m] ======  
