function [recon_img, true_img, g] = load_data_given_index_mf(mf,index)
% LOAD_DATA_GIVEN_INDEX_MF ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 04-May-2017 14:43:06 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : load_data_given_index_mf.m 

fid = fopen([mf 'recon' num2str(index) '.dat'],'rb');
recon_img = fread(fid,'float');
fclose(fid);

fid = fopen([mf 'img' num2str(index) '.dat'],'rb');
true_img = fread(fid,'float');
fclose(fid);

fid = fopen([mf 'measdata' num2str(index) '.dat'],'rb');
g = fread(fid,'float');
fclose(fid);






% ===== EOF ====== [load_data_given_index_mf.m] ======  
