function generate_v12_Analytical_Noise_TV_Projection(theta,noise,output_dirname)
% GENERATE_V12_ANALYTICAL_NOISE_TV_PROJECTION ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 14-Jun-2017 17:01:40 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : generate_v12_Analytical_Noise_TV_Projection.m 


%% Do this once to add necessary folders to path
addpath ../fista-matlab

% nohup matlab -nodesktop -nodisplay -nosplash -r generate_v7_dataset_bk &

%% Load H
% clear

nrays = 256;
% theta120 = (-60:59)*pi/180;
% [~, H120] = calc_projs(ones(256,256), theta120, nrays);
% theta = (-70:69)*pi/180;
tic;
% [~, H120_large] = calc_projs(ones(nrays,nrays), theta, nrays);
% nrays = nrays/2;
[~, H120_small] = calc_projs(ones(nrays,nrays), theta, nrays);
toc;

%write_system_matrix(H120, '/home/dlshare/xray-limitedview/data/system-matrix/H60v3');

%% Generate samples verbosely!
if 1==0
    %%
    for i=1:2
        figure;
        set(gcf,'Position',[20 112 1878 847]);
    n = 1;
    output_dirname = 'tmp/';
    noise=0;
    generate_samples_v5_Analytical_Noise(n, output_dirname, theta,H120_small,noise, 1); 
    %generate_G_TV_PSNR_dataset(n, output_dirname, H120, 1);
%     subplot(2,4,1);
%     title(['Original Image.  Theta:' num2str(60)]);
    suptitle(['Theta: ' num2str(length(theta))]);
    end;
    
end;

%% Prepare parallel workers

poolobj = gcp('nocreate');
delete(poolobj);
% Somehow check if we already have a parpool going...

num_workers=5; 
parpool('local',num_workers);
% makedir dataset_v7_120/

%% Generate Samples!
num_images=10000;
n = floor(num_images/num_workers);
parfor ix = 1:num_workers
    
%     output_dirname = 'dataset_v37_60D_Nonneg_NonIC_noNoise/';
%     noise=0;
%     generate_samplesv2(n, output_dirname, H120, n*(ix-1)+1); 
    generate_samples_v7_Analytical_Noise_TV_Projection(n, output_dirname, theta,H120_small,noise,n*(ix-1)+1+0*n*num_workers); 
end





 
% ===== EOF ====== [generate_v12_Analytical_Noise_TV_Projection.m] ======  
