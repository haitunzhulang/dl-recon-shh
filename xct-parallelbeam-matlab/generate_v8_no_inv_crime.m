function generate_v8_no_inv_crime()
% GENERATE_V8_NO_INV_CRIME ... 
%  
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 08-Mar-2017 17:04:43 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : generate_v8_no_inv_crime.m 

%% Do this once to add necessary folders to path
addpath ../fista-matlab

% nohup matlab -nodesktop -nodisplay -nosplash -r generate_v7_dataset_bk &

%% Load H
clear
% prefix = '../dlshare/xray-limitedview/data/system-matrix/';
% icols_f = [prefix, 'H120v3_icols.dat'];
% irows_f = [prefix, 'H120v3_irows.dat'];
% vals_f = [prefix, 'H120v3_vals.dat'];

% fid = fopen(icols_f,'r');
% icols = fread(fid,'float');
% fclose(fid);
% fid = fopen(irows_f,'r');
% irows = fread(fid,'float');
% fid = fopen(vals_f,'r');
% vals = fread(fid,'float');
% fclose(fid);
% H = sparse(icols,irows,vals);

nrays = 512;
% theta120 = (-60:59)*pi/180;
% [~, H120] = calc_projs(ones(256,256), theta120, nrays);
theta120 = (-50:49)*pi/180;
tic;
% [~, H120_large] = calc_projs(ones(nrays,nrays), theta120, nrays);
nrays = nrays/2;
[~, H120_small] = calc_projs(ones(nrays,nrays), theta120, nrays);
toc;

write_system_matrix(H120_small, '/home/bmkelly/xct-parallelbeam-matlab/system-matrix/H100v3');

%% Generate samples verbosely!
if 1==0
    %%
    n = 1;
    output_dirname = 'tmp/';
    generate_samplesv3_no_inv_crime(n, output_dirname, H120_large,H120_small, 1); 
    %generate_G_TV_PSNR_dataset(n, output_dirname, H120, 1);
    subplot(2,4,1);
    title(['Original Image.  Theta:' num2str(60)]);
    
end;

%% Prepare parallel workers

poolobj = gcp('nocreate');
delete(poolobj);
% Somehow check if we already have a parpool going...

num_workers=6; 
parpool('local',num_workers);
% makedir dataset_v7_120/

%% Generate Samples!

parfor ix = 1:num_workers

    n = 1500;
    output_dirname = 'dataset_v22_140_noRI_scale_nonneg_noInvCrime/';
%     generate_samplesv2(n, output_dirname, H120, n*(ix-1)+1); 
    generate_samplesv3_no_inv_crime(n, output_dirname, H120_large,H120_small, n*(ix-1)+1+1*n*num_workers);
end

%% Examine samples!

if 1==0
    %%
    PHANTOM_FILENAME_PREFIX = 'img';
    MEASDATA_FILENAME_PREFIX = 'measdata';
    RECON_FILENAME_PREFIX = 'recon';
    FILENAME_SUFFIX = '.dat';
    output_dirname = 'dataset_v7_120/';
    for i=1:100
        recon_f = [output_dirname RECON_FILENAME_PREFIX num2str(i) FILENAME_SUFFIX];
        fid = fopen(recon_f,'r'); vals = fread(fid,'float'); fclose(fid);
        recon_img = reshape(vals,[256,256]);
        
        true_f = [output_dirname PHANTOM_FILENAME_PREFIX num2str(i) FILENAME_SUFFIX];
        fid = fopen(true_f,'r'); vals = fread(fid,'float'); fclose(fid);
        true_img = reshape(vals,[256,256]);
        
        clf;
        subplot(1,2,1);
        imagesc(true_img);
        title('True img');
        subplot(1,2,2);
        imagesc(recon_img);
        title('Recon img');
        pause();
        
        
        
        
        
    end;
    
    
end;














% EoF





% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [generate_v8_no_inv_crime.m] ======  
