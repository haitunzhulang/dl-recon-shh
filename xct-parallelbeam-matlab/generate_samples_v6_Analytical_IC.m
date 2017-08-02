function generate_samples_v6_Analytical_IC(n, output_dirname, theta,H_small, noise,start_index,MP)
% GENERATE_SAMPLES_V5_ANALYTICAL_IC ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 25-Apr-2017 16:00:14 
%% Revision : 1.00 
%% Developed : 9.0.0.341360 (R2016a) 
%% Filename  : generate_samples_v5_Analytical_IC.m 
%% Constants
NX = 256; 
NY = 256;
VERBOSE = 0;
VERBOSE2 = 0;
PRINT_RATE = 100;

% tic;

MIN_ELLIPSES = 3;
MAX_NELLIPSES = 8;

% Parameters for gaussian blur filter
SMOOTH_FILTER_SIZE = [9, 9];
SMOOTH_FILTER_WIDTH = 0.75;

STEP_SIZE = 0.5;
TV_param = 0;
cost_cutoff_min = .001;
cost_cutoff_max = .001;

PHANTOM_FILENAME_PREFIX = 'img';
MEASDATA_FILENAME_PREFIX = 'measdata';
RECON_FILENAME_PREFIX = 'recon';
MASK_FILENAME_PREFIX = 'mask';
FILENAME_SUFFIX = '.dat';

% Set default values for optional parameters
if (~exist('start_index', 'var') || isempty(start_index))
    start_index = 1;
end

%% Check inputs
if (~isscalar(n) || floor(n) ~= n || n <= 0)
    error('The number of samples should be a positive integer.');
end

if (~isscalar(start_index) || floor(start_index) ~= start_index || ...
        start_index <= 0)
    error('The start index should be a positive integer.');
end

% Check if output directory exists.
if (~isdir(output_dirname))
    error('Output directory should already exist.');
end



%% Generate samples
% Initialize random number generator
% rng(sum(100*clock));
% rng(1337);
% rand() returns numbers on the open interval (0, 1) according to a uniform
% distribution

data.H = H_small;
h = fspecial('gaussian', SMOOTH_FILTER_SIZE, SMOOTH_FILTER_WIDTH);

for i = start_index:(n+start_index-1)
%     tic;
    if (mod(i-1, PRINT_RATE) == 0)
        fprintf('%s: Sample %d\n', datestr(now), i);
    end
    % Create image
%     num_ellipses = round((MAX_NELLIPSES-MIN_ELLIPSES)*rand() + MIN_ELLIPSES);
    
    [img,g,ellipses] = load_sim_projection(i,NX,theta);
    
%     [img,g] = simulate_projection(NX, theta,num_ellipses);
    
%     img = imfilter(img, h, 'replicate');
    
    if (VERBOSE) || (VERBOSE2)
        clf;
        subplot(2,4,1);
        imagesc(img);
        title('Original image');
        colorbar;
        pause(.01);
    end
    
    
    
%     keyboard();
    g = H_small*img(:);

    g_noise=g;
    
    g = g(:);
    
    data.g = g_noise(:);
    
    
    cost_cutoff= (cost_cutoff_max-cost_cutoff_min)*(rand(1))+cost_cutoff_min;
%     tic;
    img_recon = MP*reshape(img,[256*256 1]);
%     toc;
    img_recon = reshape(img_recon,[256 256]);
%     img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
%             STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
%             'min_rel_cost_diff', cost_cutoff,'max_iter',2000);

    if VERBOSE
        data.g = g(:);
%         noiseless_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
%                 STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
%                 'min_rel_cost_diff', cost_cutoff,'max_iter',1000);
        noiseless_recon = img_recon;
    end;
        
    
    % Recon
    fid = fopen([output_dirname, filesep, RECON_FILENAME_PREFIX, ...
    num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, img_recon, 'float');
    fclose(fid);

    % Mask
%     fid = fopen([output_dirname, filesep, MASK_FILENAME_PREFIX, ...
%     num2str(i-1), FILENAME_SUFFIX], 'wb');
%     fwrite(fid, img_mask, 'float');
%     fclose(fid);

    % True Img
    fid = fopen([output_dirname, filesep, PHANTOM_FILENAME_PREFIX, ...
        num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, img, 'float');
    fclose(fid);

    % Measurement data
    fid = fopen([output_dirname, filesep, MEASDATA_FILENAME_PREFIX, ...
        num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, g, 'float');
    fclose(fid);
    
       
    % Different stopping criteria
    if VERBOSE2
        Legend = {}; Legend{end+1} = 'True Image';
        ccs = [.05,.03,.01,.005,.001,.0001];
        x_vert = 150;
        y_profiles = zeros(7,256);
        tmp = imresize(img,.5);
        y_profiles(1,:) = tmp(:,x_vert);
        subplot(2,4,1);
        title('Original Image');
        for i=length(ccs):-1:1
            subplot(2,4,1+i+1);
            ir = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', ccs(i),'max_iter',1000);
            imagesc(ir);
            title(['Recon stopping threshold:' num2str(ccs(i))]);
            colorbar;
            y_profiles(i+1,:) = ir(:,x_vert);
            Legend{end+1} = ['Cutoff: ' num2str(ccs(i))];
           
            drawnow();
            pause(0.001);
        end;
        
        subplot(2,4,2);
        hold on;
        plot(y_profiles');
        legend(Legend);
        title(['Profile Plot, x vert: ' num2str(x_vert)]);
        
        return;
        
    end;
    
    if VERBOSE
%         data.g = g_orig(:);
%         img_recon_noiseless = fistatv2d(@cost_func_xray_H, zeros(NX/2,NY/2), data, ...
%             STEP_SIZE, 0, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
%             'min_rel_cost_diff', cost_cutoff,'max_iter',1000, 'proj_op','nonneg');
        Legend = {}; Legend{end+1} = 'True Image';Legend{end+1} = 'Recon Image';
%         subplot(2,4,1);
%         title('Original Image');
%         subplot(2,4,2);
%         imagesc(img_recon);
%         colorbar;
%         title(['Noisy Recon. P_noise: ' num2str(P_noise) ', lambda: ' num2str(lamb)]);
%         subplot(2,4,3);
%         imagesc(img_recon);
%         colorbar;
%         title('Noisy Recon');

        subplot(2,4,3);
        imagesc(img-img_recon);
        title('Original Image - Recon Image');
        colorbar;
        
        
        subplot(2,4,4);
        imagesc(noiseless_recon);
        colorbar;
        title('Noiseless Recon');
        
%         subplot(2,4,7);
%         imagesc(g_noise);
%         colorbar;
%         title('Noisy G');
        
        subplot(2,4,8);
        imagesc(reshape(g,size(g_noise)));
        colorbar;
        title('Noiseless G');
        
        
        x_vert = 150;
        y_profiles = zeros(7,256);
%         tmp = imresize(img,.5);
        y_profiles(1,:) = img(:,x_vert);
        y_profiles(2,:) = img_recon(:,x_vert);
        
        
        reg_params =[];%[0.005,.001,.0005,.0001,.00005];
        for i=1:length(reg_params)
            subplot(2,4,4+i);
            ir = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, reg_params(i), 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cost_cutoff,'max_iter',200);
            imagesc(ir);
            title(['TV Recon, TV:' num2str(reg_params(i))]);
            colorbar;
            y_profiles(i+2,:) = ir(:,x_vert);
            Legend{end+1} = ['TV: ' num2str(reg_params(i))];
            drawnow();
            pause(0.001);
        end;        
        
        subplot(2,4,2);
        hold on;
        plot(y_profiles');
        legend(Legend);
        title(['Profile Plot, x vert: ' num2str(x_vert)]);
        
    end;
    
    
        
%     toc;
end

% toc;
% ===== EOF ====== [generate_samples_v5_Analytical_IC.m] ======  
