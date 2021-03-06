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

nrays = 256;
% theta120 = (-60:59)*pi/180;
% [~, H120] = calc_projs(ones(256,256), theta120, nrays);
theta120 = (-30:29)*pi/180;
tic;
[~, H120] = calc_projs(ones(256,256), theta120, nrays);
toc;

%% Generate samples verbosely!
if 1==0
    %%
    n = 1;
    output_dirname = 'tmp/';
    realistic_generate_samples(n, output_dirname, H120, 50); 
    title(['Theta: ' num2str(60)]);
    
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
    output_dirname = 'dataset_v14_60_stochastic_noRI_nonneg/';
    generate_samplesv2(n, output_dirname, H120, n*(ix-1)+1); 
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