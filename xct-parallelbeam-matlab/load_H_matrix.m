function H= load_H_matrix(prefix_file_path,theta)
% LOAD_H_MATRIX ... 
%  
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 09-May-2017 16:11:22 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : load_H_matrix.m 

% prefix_file_path = 'system-matrix/H60v3_';
% disp([prefix_file_path 'vals.dat']);
pid = fopen([prefix_file_path 'vals.dat'],'rb');
vals = fread(pid,'float');
fclose(pid);

pid = fopen([prefix_file_path 'icols.dat'],'rb');
icols = fread(pid,'float');
fclose(pid);

pid = fopen([prefix_file_path 'irows.dat'],'rb');
irows = fread(pid,'float');
fclose(pid);

H = sparse(irows,icols,vals,theta*256,256*256);




% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [load_H_matrix.m] ======  
