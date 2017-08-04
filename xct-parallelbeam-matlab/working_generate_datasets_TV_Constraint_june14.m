function working_generate_datasets_TV_Constraint_june14()
% WORKING_GENERATE_DATASETS_TV_CONSTRAINT_JUNE14 ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 14-Jun-2017 17:16:34 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : working_generate_datasets_TV_Constraint_june14.m 

% add the toolbox_image
addpath(genpath('./toolbox_image'))
%output_dirname = 'dataset_v5_50D_IC_Noise_TVC/';mkdir(output_dirname);
output_dirname = 'dataset_v5_60D_IC_Noise_TVC/';mkdir(output_dirname);
noise=.02;
theta = (-20:29)*pi/180;
generate_v12_Analytical_Noise_TV_Projection(theta,noise,output_dirname);





 
% ===== EOF ====== [working_generate_datasets_TV_Constraint_june14.m] ======  
