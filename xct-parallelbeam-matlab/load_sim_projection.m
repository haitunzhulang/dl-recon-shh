function[img,g,ellipses] = load_sim_projection(index,NX,theta)
% LOAD_SIM_PROJECTION ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 30-May-2017 16:52:45 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : load_sim_projection.m 

mf ='images_and_measured_data_for_all/';
fname = [mf num2str(index) '.mat'];

load(fname);

[img,g,ellipses] = simulate_projection(NX,theta,0,ellipses);



 
% ===== EOF ====== [load_sim_projection.m] ======  
