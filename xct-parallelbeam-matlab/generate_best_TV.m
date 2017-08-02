function generate_best_TV(theta,mf,tv_storage,nonneg)
% GENERATE_BEST_TV ... 
%  
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 02-May-2017 17:42:36 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : generate_best_TV.m 

addpath ../fista-matlab

%%
nrays = 256;
% theta = (-30:29)*pi/180;
tic;
[~, H_real] = calc_projs(ones(nrays,nrays), theta, nrays);
toc;

%%

% v37 dataset_v37_60D_Nonneg_NonIC_noNoise
% mf = 'dataset_v37_60D_Nonneg_NonIC_noNoise/';
% tv_storage = 'tmp_tv_v37/';
% 
% % v38
% mf = 'dataset_v38_100D_Nonneg_NonIC_noNoise/';
% tv_storage = 'tmp_tv_v38/';

%%

if ~exist('nonneg')
	nonneg=1;
end;
	

b=40;
support = [1+b,256-b,1+b,256-b];

for i=7501:9000
    fid = fopen([mf 'measdata' num2str(i), '.dat'], 'rb');
    g = fread(fid,'float');
    fclose(fid);
    
    fid = fopen([mf 'img' num2str(i), '.dat'], 'rb');
    f_true = fread(fid,'float');
    fclose(fid);
    f_true = reshape(f_true,[256 256]);
    
    acquire_best_TV(g,f_true,H_real,support,i,tv_storage,nonneg);
end;






% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [generate_best_TV.m] ======  
