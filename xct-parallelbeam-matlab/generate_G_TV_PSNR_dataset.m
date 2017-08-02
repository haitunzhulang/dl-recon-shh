function generate_samplesv2(n, output_dirname, H, start_index)
% GENERATE_SAMPLESV2 Generate random samples for training and testing
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
NPARAMETERS = 6; % Number of parameters needed to define an ellipse
VERBOSE = 0;
PRINT_RATE = 100;

TV_Range = log10([.1, .00001]);

NOISE_SCALE = 0;
MAX_ITER = 300;

SCALE = 1;
MIN_ELLIPSES = 1;
MAX_NELLIPSES = 5;

% All values are not inclusive
MIN_CENTER = -0.3;
MAX_CENTER = 0.3;
MIN_AXES_LEN = 0.02; 
MAX_AXES_LEN = 0.3;
MIN_ROT_ANGLE = 0;
MAX_ROT_ANGLE = 2*pi;
MIN_INTENS = 0.4;
MAX_INTENS = 1;

% Parameters for gaussian blur filter
SMOOTH_FILTER_SIZE = [9, 9];
SMOOTH_FILTER_WIDTH = 0.75;

STEP_SIZE = 0.75;

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

if (~ismatrix(H) || size(H,2) ~= NX*NY)
    error(['H should be a matrix whose number of columns is equal to ', ...
        'the number of elements in the sample image.']);
end

%% Generate samples
% Initialize random number generator
% rng(sum(100*clock));
rng(1336);
% rand() returns numbers on the open interval (0, 1) according to a uniform
% distribution

data.H = H;

if (VERBOSE)
    figure;
end

h = fspecial('gaussian', SMOOTH_FILTER_SIZE, SMOOTH_FILTER_WIDTH);

for i = start_index:(n+start_index-1)
    if (mod(i-1, PRINT_RATE) == 0)
        fprintf('%s: Sample %d\n', datestr(now), i);
    end
    % Create image
    nellipses = round((MAX_NELLIPSES-MIN_ELLIPSES)*rand() + MIN_ELLIPSES);
    ellipses = rand(nellipses, NPARAMETERS);
    
    % Adjust ellipses so that the parameters is in the specified range
    ellipses(:,1:2) = (MAX_CENTER - MIN_CENTER)*ellipses(:,1:2) + ...
        MIN_CENTER;
    ellipses(:,3:4) = (MAX_AXES_LEN - MIN_AXES_LEN)*ellipses(:,3:4) + ...
        MIN_AXES_LEN;
    ellipses(:,5) = (MAX_ROT_ANGLE - MIN_ROT_ANGLE)*ellipses(:,5) + ...
        MIN_ROT_ANGLE;
    ellipses(:,6) = (MAX_INTENS - MIN_INTENS)*ellipses(:,6) + MIN_INTENS;

    
    img = sim_image(NX, ellipses, SCALE);
    % Rounding to nearest decimal
    img = round(img*10)/10;
    % Smooth?
%     img = imfilter(img, h, 'replicate');
    
    img_mask = sim_image(NX, ellipses(1,:), SCALE);
    img_mask = imfilter(img_mask, h, 'replicate');
    
    if (VERBOSE)
        clf;
        subplot(1,4,1);
        imagesc(img);
        title('True img');
        colorbar;
        pause(.01);
    end
    
    fid = fopen([output_dirname, filesep, MASK_FILENAME_PREFIX, ...
        num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, img_mask, 'float');
    fclose(fid);
    
    fid = fopen([output_dirname, filesep, PHANTOM_FILENAME_PREFIX, ...
        num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, img, 'float');
    fclose(fid);
    
    g = H*img(:);
    
    fid = fopen([output_dirname, filesep, MEASDATA_FILENAME_PREFIX, ...
        num2str(i-1), FILENAME_SUFFIX], 'wb');
    fwrite(fid, g, 'float');
    fclose(fid);
    
    
    
    
    data.g = g;
    reg_param = 10^(TV_Range(2) + (TV_Range(1)-TV_Range(2))*rand(1));
    
    if (VERBOSE)
        img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
        STEP_SIZE, reg_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
        'max_iter', MAX_ITER, 'proj_op','nonneg');
    
        subplot(1,4,2);
        imagesc(img_recon);
        title(['Recon img - TV Weight: ' num2str(reg_param) '. PSNR: ' num2str(psnr(img,img_recon))]);
        colorbar;
        return;
    end;
    
    
    
    img_recon = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
        STEP_SIZE, reg_param, 'output_filename_prefix', '', 'verbose', VERBOSE, ...
        'max_iter', MAX_ITER, 'proj_op','nonneg');
    
    
    
    
    save_reg_param = log10(reg_param)/(TV_Range(2) - TV_Range(1));
    
    fid = fopen([output_dirname, filesep, RECON_FILENAME_PREFIX, ...
        num2str(i-1), '_',num2str(save_reg_param), FILENAME_SUFFIX], 'wb');
    fwrite(fid, img_recon, 'float');
    fclose(fid);
end




