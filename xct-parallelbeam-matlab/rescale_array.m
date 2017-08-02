function array =  rescale_array(array,nmin,nmax)
% RESCALE_ARRAY ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 31-May-2017 18:26:49 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : rescale_array.m 


array = nmin + (nmax-nmin)*(array - min(array(:)))./(max(array(:)) - min(array(:)));




 
% ===== EOF ====== [rescale_array.m] ======  
