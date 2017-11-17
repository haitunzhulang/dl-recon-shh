function working_generate_datasets_noIC_Noise_August9()
% WORKING_GENERATE_DATASETS_NO_IC_NOISE_AUG9 ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 9-Aug-2017 2:02 pm
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : working_generate_datasets_noIC_Noise_August9.m

% add the toolbox_image
addpath(genpath('./toolbox_image'))
%output_dirname = 'dataset_v5_50D_IC_Noise_TVC/';mkdir(output_dirname);
output_dirname = 'dataset_v5_100D_noIC_Noise/';mkdir(output_dirname);
noise=.02;
theta = (-50:49)*pi/180;
tv_param=0;
generate_v10_Analytical_Noise(theta,noise,output_dirname,tv_param);



% ===== EOF ====== [working_generate_datasets_TV_Constraint_june14.m] ======  
