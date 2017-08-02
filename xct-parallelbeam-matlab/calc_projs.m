function [projs, H] = calc_projs(im, theta, nrays)
% CALC_PROJS Calculates the projection data for parallel beam X-ray CT
% INPUTS:
%   im - a n x n image to compute the projection data
%   theta - a 1 x nprojs array with the angle of each projection [rad.]
%   nrays [OPT] - the number of detectors per projection angle; the
%       detectors are evenly distributed along the length of the image (not
%       its diagonal). [Default: n]
% OUTPUTS:
%   projs - a nrays x nprojs matrix corresponding to the measused
%       projection data (sinogram)
%   H [opt] - a sparse (nrays*nprojs) x (n*n) matrix corresponding to the
%       system matrix of the specified system. This matrix can be used to
%       compute the projection data as H*im(:). Having an explicit form for
%       H can be useful for small-scale optimization problems. Storing H
%       will dramatically slow down this function.
% Examples:
%   projs = calc_projs(img, (0:179)*pi/180);
%   projs = calc_projs(img, (0:179)*pi/180, 256);
%   [projs, H] = calc_projs(img, (0:179)*pi/180, 256);
% See also: sim_image

% profile on

if (size(im,1) ~= size(im,2))
    error('The input image must be square.');
end
n = size(im,1);

if (~exist('nrays', 'var') || isempty(nrays))
    nrays = n;
end
nprojs = length(theta);
projs = zeros(nrays, nprojs);

% Store vals and indices of all non-zero values in H as three separate 
% arrays and then construct H outside of loops
if (nargout >= 2)
    irows = [];
    icols = [];
    vals = [];
%     H = sparse(nrays*nproj, numel(im));
end

% Create matrices of the x and y axis values over the image
x = linspace(-1, 1, n);
x = repmat(x, n, 1);
y = linspace(1, -1, n)';
y = repmat(y, 1, n);
% Assume that the object is contained within the circle inscribed within
% the image. Do not place rays which are exactly tangent to this circle as
% the intersection of these rays with the circle will have length 0 and 
% will not contribute to the projection data.
dx = abs(x(1,1)-x(1,2));
dtshift = min([dx/2, n/nrays*dx/2]);
t = linspace(-1 + dtshift, 1 - dtshift, nrays);

% Linearize the image
im = reshape(im, numel(im), 1);

% hwait = waitbar(0, 'Generating projections...');
for i = 1:length(theta)
%     waitbar(i/length(theta), hwait);
    for j = 1:length(t)
        w = calc_weights_interp(theta, x, y, t, (i-1)*length(t)+j);
        projs(j,i) = w'*im;
        if (nargout >= 2)
            inds = find(abs(w) > 0);
%             H((i-1)*nrays+(j-1)+1,inds) = w(inds)';
            irows = [irows; ((i-1)*nrays+(j-1)+1)*ones(length(inds),1)];
            icols = [icols; inds];
            vals = [vals; w(inds)];
        end
    end
end
% close(hwait);

if (nargout >= 2)
    H = sparse(irows, icols, vals, nrays*nprojs, numel(im));
end

% profile viewer

