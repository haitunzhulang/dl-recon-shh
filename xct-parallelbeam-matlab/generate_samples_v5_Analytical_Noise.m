function generate_samples_v5_Analytical_Noise(n, output_dirname, theta,H_small, noise,start_index,tv_param)
% GENERATE_SAMPLESV3 Generate random samples for training and testing
% INPUTS:
%   n - the total number of samples to generate
%   output_dirname - name of the directory where samples will be stored
%   H - m x k (sparse) matrix that produces the measured data from a given
%       samples
%   start_index [OPT] - index to give to the first sample
% OUTPUTS:
%   A collection of binary single-precision floating point data files will
%   be written to disk in the specified directory. For each sample, three
%   files will be written:
%      (1) True phantom (img#.dat)
%      (2) Measured data (measdata#.dat)
%      (3) Reconstructed image (recon#.dat)
% Examples:
%   generate_samplesv2(1000, 'Samples', H120v2); % Indexed from 0
%   generate_samplesv2(1000, 'Samples', H120v2, 1000); % Indexed from 1000
% See also:

%% Constants
NX = 256; 
NY = 256;
VERBOSE = 0;
VERBOSE2 = 0;
PRINT_RATE = 100;

MIN_ELLIPSES = 3;
MAX_NELLIPSES = 8;

% Parameters for gaussian blur filter
SMOOTH_FILTER_SIZE = [9, 9];
SMOOTH_FILTER_WIDTH = 0.75;

STEP_SIZE = 0.5;
TV_param = tv_param;
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
%    error('Output directory should already exist.');
	mkdir(output_dirname)
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
%     g = H_large*img(:);
%     g=reshape(g,[NX length(g)/NX]);
%     g = g(1:2:end,:);
    
    % Add gaussian noise
%     g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
    g_noise = imnoise(g,'gaussian',0,(noise*max(g(:)))^2);
%     g_noise=g;
    
    g = g(:);
    
    data.g = g_noise(:);
    
    
    cost_cutoff= .001;
    img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, TV_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
            'min_rel_cost_diff', cost_cutoff,'max_iter',1000,'proj_op','nonneg');

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
        
        
        reg_params =[.001,.0001];%[0.005,.001,.0005,.0001,.00005];
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
    
    
        
    
end

