function projs = calc_projs_ell(ellipses, theta, nrays)
% CALC_PROJS_ELL Calculate the projections of an image made from ellipses
% INPUTS:
%   ellipses - matrix of ellipse parameters; see sim_image documentation.
%   theta - a 1 x nprojs array with the angle of each projection [rad.]
%   nrays - the number of detectors per projection angle; the detectors are
%       evenly distributed along the length of the image (not its diagonal). 
% OUTPTS:
%   projs - a nrays x nprojs matrix of the projection data
% Examples:
%   [~, ellipses] = sim_image(256);
%   projs = calc_projs_ell(ellipses, (0:179)*pi/180, 256);
% See also: sim_image

% The number of points in each projection is chosen in order to maintain 
% compatibility with MATLAB's built-in radon, iradon functions.
% if (~exist('nrays', 'var') || isempty(nrays))
%     nrays = 2*ceil(norm(size(im)-floor((size(im)-1)/2)-1))+3;
% end
nprojs = length(theta);
projs = zeros(nrays, nprojs);

% The ellipses are defined based on a coordinate system where both x and y
% range from -1 to 1. Thus, t can range from -sqrt(2) to sqrt(2).
t = linspace(-1, 1, nrays)';
for i = 1:length(theta)
	for j = 1:size(ellipses, 1)
        [x0, y0, A, B, alpha, rho] = extract_ellipse_pars(ellipses, j); 
        % Calculate radon transform of ellipse REF: CTI Chapter 3.1
		s = sqrt(x0^2 + y0^2);
		gamma = atan2(y0, x0);
		tadj = t - s*cos(gamma - theta(i)); 
        new_theta = theta(i) - alpha;
		a2 = A^2*cos(new_theta)^2 + B^2*sin(new_theta)^2;
        inds = abs(tadj)<=sqrt(a2); % Find non-zero indices
        % The contributions from each ellipse can be summed up
        % independently.
		projs(inds,i) = projs(inds,i) + 2*rho*A*B/a2*sqrt(a2-tadj(inds).^2);
	end
end	

