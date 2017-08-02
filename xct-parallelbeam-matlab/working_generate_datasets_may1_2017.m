function working_generate_datasets_may1_2017()
% WORKING_GENERATE_DATASETS_MAY1_2017 ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 01-May-2017 17:49:24 
%% Revision : 1.00 
%% Developed : 9.0.0.341360 (R2016a) 
%% Filename  : working_generate_datasets_may1_2017.m 

% v0
output_dirname = 'dataset_v0_50D_IC_noNoise/';mkdir(output_dirname);
noise=0;
theta = (-25:24)*pi/180;
generate_v10_Analytical_Noise(theta,noise,output_dirname);

% v1
tv_params=[0 ];%.0001 .0005 .001 .005];
for i=1:length(tv_params)
    output_dirname = ['dataset_v1_50D_Nonneg_NonIC_noNoise_' num2str(tv_params(i)) '/'];
    mkdir(output_dirname);
    noise=0;
    theta = (-25:24)*pi/180;
    generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_params(i));
end;

% v2
tv_params=[0 .0001 .0005 .001 .005];
for i=1:length(tv_params)
    output_dirname = ['dataset_v2_70D_Nonneg_NonIC_noNoise_' num2str(tv_params(i)) '/'];
    mkdir(output_dirname);
    noise=0;
    theta = (-35:34)*pi/180;
    generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_params(i));
    
end;

% v3
output_dirname = 'dataset_v3_90D_Nonneg_NonIC_noNoise/';mkdir(output_dirname);
noise=0;
theta = (-45:44)*pi/180;
generate_v10_Analytical_Noise(theta,noise,output_dirname);


% v4
tv_params=[0 ];%.0001 .0005 .001 .005];
for i=1:length(tv_params)
    output_dirname = ['dataset_v4_50D_Nonneg_NonIC_Noise_' num2str(tv_params(i)) '/'];
    mkdir(output_dirname);
    noise=.02;
    theta = (-25:24)*pi/180;
    generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_params(i));
end;

% v5
output_dirname = 'dataset_v5_70D_Nonneg_NonIC_Noise/';
mkdir(output_dirname);
noise=.02;
theta = (-35:34)*pi/180;
generate_v10_Analytical_Noise(theta,noise,output_dirname);

% v6
output_dirname = 'dataset_v6_90D_Nonneg_NonIC_Noise/';
mkdir(output_dirname);
noise=.02;
theta = (-45:44)*pi/180;
generate_v10_Analytical_Noise(theta,noise,output_dirname);







% ===== EOF ====== [working_generate_datasets_may1_2017.m] ======  
