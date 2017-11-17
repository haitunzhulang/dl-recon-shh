function generate_v9_noIC_Noise()
% GENERATE_V9_NOIC_NOISE ... 
%  Same as generate_v8 except with noise
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 05-Apr-2017 18:06:54 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.0.0.341360 (R2016a) 
%% FILENAME  : generate_v9_noIC_Noise.m 


%% Do this once to add necessary folders to path
addpath ../fista-matlab

% nohup matlab -nodesktop -nodisplay -nosplash -r generate_v7_dataset_bk &

%% Load H
clear

nrays = 512;
% theta120 = (-60:59)*pi/180;
% [~, H120] = calc_projs(ones(256,256), theta120, nrays);
theta120 = (-10:9)*pi/180;
tic;
[~, H120_large] = calc_projs(ones(nrays,nrays), theta120, nrays);
nrays = nrays/2;
[~, H120_small] = calc_projs(ones(nrays,nrays), theta120, nrays);
toc;

%write_system_matrix(H120, '/home/dlshare/xray-limitedview/data/system-matrix/H60v3');

%% Generate samples verbosely!
if 1==0
    %%
    n = 1;
    output_dirname = 'tmp/';
    noise=0;
    generate_samples_v4_noIC_noise(n, output_dirname, H120_large,H120_small,noise, 1); 
    %generate_G_TV_PSNR_dataset(n, output_dirname, H120, 1);
%     subplot(2,4,1);
%     title(['Original Image.  Theta:' num2str(60)]);
    
end;

%% Prepare parallel workers

poolobj = gcp('nocreate');
delete(poolobj);
% Somehow check if we already have a parpool going...

num_workers=5; 
parpool('local',num_workers);
% makedir dataset_v7_120/

%% Generate Samples!

parfor ix = 1:num_workers
    n = 1000;
    output_dirname = 'dataset_v30_20_noIC/';
    noise=.05;
%     generate_samplesv2(n, output_dirname, H120, n*(ix-1)+1); 
    generate_samples_v4_noIC_noise(n, output_dirname, H120_large,H120_small,noise,n*(ix-1)+1+1*n*num_workers);
end




% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [generate_v9_noIC_Noise.m] ======  
