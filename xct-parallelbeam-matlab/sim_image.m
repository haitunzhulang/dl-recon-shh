function [im, ellipses] = sim_image(n, ellipses, scale)
% SIM_IMAGE Creates an nxn image consisting of a collection of ellipses
% INPUTS:
%	n - number of rows and columns in image
%	ellipses [OPT] - a kxl matrix, where each row represents a separate
%		ellipse and the columns are as follows: [x y A B alpha rho], where
%		the ellipse is given by x^2/A^2 + y^2/B^2 <= 1, with center (x, y)
%		and rotated by alpha with intensity rho. Note that the intensities
%		of overlapping ellipses are additive. The top left of the image is
%       treated as (-1, 1) and the bottom right is(1, -1). 
%       Default: High-contrast Shepp-Logan phantom
%   scale [OPT] - if true, scale the image so that the intensities are
%       between 0 and 1; if false, the image will not be scaled and the
%       caller is responsible for ensuring that the intensities are within
%       the proper range. Default: 1
% OUTPUTS:
%	im - a nxn image with a black background and the requested ellipses;
%		units of the image are double
%   ellipses - matrix of the ellipses used to generate the image
% Examples:
% 	im = sim_image(512, [0 0 0.25 0.25 0 0.6]);
% See also:

if (~exist('ellipses', 'var') || isempty(ellipses))
    % Higher contrast version of Shepp-Logan phantom; values, other than
    % intensities, taken from CTI Table 3.1. This is similar to the image 
    % obtained from the built-in phantom function.
    ellipses = [0 0 0.92 0.69 pi/2 2.0;
                0 -0.0184 0.874 0.6624 pi/2 -1.6;
                0.22 0 0.31 0.11 72*pi/180 -0.4;
                -0.22 0 0.41 0.16 108*pi/180 -0.4;
                0 0.35 0.25 0.21 pi/2 0.15;
                0 0.1 0.046 0.046 0 0.15;
                0 -0.1 0.046 0.046 0 0.15;
                -0.08 -0.605 0.046 0.023 0 0.15;
                0 -0.605 0.023 0.023 0 0.15;
                0.06 -0.605 0.046 0.023 pi/2 0.15];
end

if (~exist('scale', 'var') || isempty(scale))
    scale = 1;
end

im = zeros(n);

% Create matrices of the x and y axis values over the image
x = linspace(-1, 1, n);
x = repmat(x, n, 1);
y = linspace(1, -1, n)';
y = repmat(y, 1, n);

% Add the ellipses to the image
for i = 1:size(ellipses, 1)
    [x0, y0, A, B, alpha, rho] = extract_ellipse_pars(ellipses, i);

    % A general equation for an ellipse can be obtained by taking the
    % standard equation for an ellipse centered at the origin with axes
    % aligning with the axes of the coordinate system 
    % (x^2/A^2 + y^2/B^2 <= 1) and rotating and translating that ellipse
    
    % Find the indices of the pixels in the interior of the ellipse
	inds = (((x-x0)*cos(alpha)+(y-y0)*sin(alpha)).^2/A^2 + ...
		((x-x0)*sin(alpha)-(y-y0)*cos(alpha)).^2/B^2 <= 1);
	im(inds) = im(inds) + rho;
end

% Scale image so that intensities are between 0 and 1
% if (scale)
%     im = (im-min(im(:)))/(max(im(:))-min(im(:)));
% end

