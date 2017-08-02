function generate_v11_Analytical_IC(theta,output_dirname)
% GENERATE_V11_ANALYTICAL_IC ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 25-Apr-2017 15:58:54 
%% Revision : 1.00 
%% Developed : 9.0.0.341360 (R2016a) 
%% Filename  : generate_v11_Analytical_IC.m 
%% Do this once to add necessary folders to path
addpath ../fista-matlab

% nohup matlab -nodesktop -nodisplay -nosplash -r generate_v7_dataset_bk &

%% Load H
clear

nrays = 256;
% theta120 = (-60:59)*pi/180;
% [~, H120] = calc_projs(ones(256,256), theta120, nrays);
theta = (-70:69)*pi/180;
tic;
% [~, H120_large] = calc_projs(ones(nrays,nrays), theta, nrays);
% nrays = nrays/2;
[~, H120_small] = calc_projs(ones(nrays,nrays), theta, nrays);
toc;

tic;
[~, ~, V, flag] = svds(H120_small,min(size(H120_small)));
toc;

tic;
MP = V*V';
toc;

save MP_140D_LV.mat MP -v7.3
%write_system_matrix(H120, '/home/dlshare/xray-limitedview/data/system-matrix/H60v3');

%% Generate samples verbosely!
if 1==0
    %%
    for i=1:2
        clf;
        set(gcf,'Position',[20 112 1878 847]);
    n = 1;
    output_dirname = 'tmp/';
    noise=0;
    generate_samples_v6_Analytical_IC(n, output_dirname, theta,H120_small,noise, 1,MP); 
    %generate_G_TV_PSNR_dataset(n, output_dirname, H120, 1);
%     subplot(2,4,1);
%     title(['Original Image.  Theta:' num2str(60)]);
    suptitle(['Theta: ' num2str(length(theta))]);
    end;
    
end;

%% Prepare parallel workers

% poolobj = gcp('nocreate');
% delete(poolobj);
% Somehow check if we already have a parpool going...

num_workers=6; 
% parpool('local',num_workers);
% makedir dataset_v7_120/

%% Generate Samples!

for ix = 1:num_workers
    n = 2000;
    output_dirname = 'dataset_v36_IC_100LV_noNoise/';
    noise=0;
%     generate_samplesv2(n, output_dirname, H120, n*(ix-1)+1); 
    generate_samples_v6_Analytical_IC(n, output_dirname, theta,H120_small,noise,n*(ix-1)+1+0*n*num_workers,MP); 
end





% ===== EOF ====== [generate_v11_Analytical_IC.m] ======  
