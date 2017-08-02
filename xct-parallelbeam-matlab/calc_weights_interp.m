function w = calc_weights_interp(theta, x, y, t, i)
% CALC_WEIGHTS_INTERP Calculates weights for each image pixel using an interpolation basis
% INPUTS:
%   theta - an array of the projection angles in radians, one element
%       should exist for each projection point in order to match the 1D
%       representation of the projection data
%   x - the x values of each image pixel
%   y - the y values of each image pixel
%   t - an array of the t-values that the projections were taken over
%   i - the index of the current projection ray
% OUTPUT:
%   w - an array of the weights for each image pixel; in this case, the
%       weight is 1 if the current ray passes through the pixel and 0
%       otherwise

% Weights wij: the weight of the jth pixel to the ith ray
w = zeros(numel(x),1);

j = mod(i-1, length(t))+1; % Find the position in t for the current ray
k = ceil(i/length(t)); % Find the angle of the current ray

ds = (x(1,2)-x(1,1))/2; % Set step along ray equal to half pixel width

L = calc_ray_length(theta, x, y, t, k, j);
% fprintf('i: %d t: %f theta: %f Ray length: %f\n', i, t(j), theta(k), L);
M = floor(L/ds); % Number of points to sample along the current ray
s = (-L/2+ds/2)+ds*(0:(M-1)); % Add ds/2 to starting position?

% fprintf('True ray length: %.4f\n', L);
% fprintf('Approx. ray length: %.4f\n', max(s) - min(s));
% fprintf('Error: %.4f\n', L - (max(s)-min(s)));
% fprintf('ds: %.4f\n', ds);
% fprintf('M: %d\n', M);
% fprintf('M*ds %.4f\n', M*ds);

max_x = max(x(:));
min_x = min(x(:));
max_y = max(y(:));
min_y = min(y(:));
dx = abs(x(1,1)-x(1,2));
dy = abs(y(1,1)-y(2,1));

nrows = size(x,1);

for m = 1:M
    % Find the (x,y) coordinates of the mth point
    xim = t(j)*cos(theta(k)) - s(m)*sin(theta(k));
    yim = t(j)*sin(theta(k)) + s(m)*cos(theta(k));
    
    % Check if the current point is outside the image
    if (xim >= max_x || xim <= min_x || yim >= max_y || yim <= min_y)
        continue;
    end

    % Find the nearest x and y points
    x1ind = floor((xim-min_x)/dx)+1;
    y2ind = floor((max_y-yim)/dy)+1;

    x1 = x(1, x1ind);
    x2 = x(1, x1ind+1);
    y1 = y(y2ind+1, 1);
    y2 = y(y2ind, 1);

    inds11 = y2ind+1 + (x1ind-1)*nrows;
    inds12 = y2ind + (x1ind-1)*nrows;
    inds21 = y2ind+1 + (x1ind+1-1)*nrows;
    inds22 = y2ind + (x1ind+1-1)*nrows;
    
    % Adjust the weights so that sum(w) = L
    if (m == 1 || m == M)
        scale = 1 + (L - M*ds)/(2*ds);
    else
        scale = 1;
    end
    denom = scale/((x2-x1)*(y2-y1));   
    w(inds11) = w(inds11) + (x2-xim)*(y2-yim)*denom;
    w(inds21) = w(inds21) + (xim-x1)*(y2-yim)*denom;
    w(inds12) = w(inds12) + (x2-xim)*(yim-y1)*denom;
    w(inds22) = w(inds22) + (xim-x1)*(yim-y1)*denom;
    
%     if (m == 1)
%         disp(sum([w(inds11), w(inds12), w(inds21), w(inds22)]))
%     end
end
w = w*ds;
% fprintf('Sum w: %.4f\n', sum(w));

function [x, y] = rotate_coords(t, s, theta)
% ROTATE_COORDS Rotates the coordinate system
% INPUTS:
%   t - old "x" coordinate
%   s - old "y" coordinate
%   theta - the angle of rotation, in radians
% OUTPUTS:
%   x - new x coordinate
%   y - new y coordinate

R = [cos(theta) -sin(theta); 
    sin(theta) cos(theta)];
pos = R*[t s]';
x = pos(1);
y = pos(2);


function r = calc_reco_radius(x, y)
% CALC_RECO_RADIUS Calculate the radius of the reconstruction region
% Choose the reconstruction region to be the largest circle that can be
% inscribed within the image coordinate space
% INPUTS:
%   x - the x values of each image pixel
%   y - the y values of each image pixel
% OUTPUTS:
%   r - the radius of the circle describing the reconstruction region

xrange = max(x(:)) - min(x(:));
yrange = max(y(:)) - min(y(:));
r = min([xrange/2, yrange/2]);

function L = calc_ray_length(theta, x, y, t, ip, i)
% CALC_RAY_LENGTH Calculates the length of the current ray
% INPUTS:
%   theta - an array of the angles that projections were made at
%   x - the x values of each image pixel
%   y - the y values of each image pixel
%   t - the values of t for a projection
%   ip - the index of the projection
%   i - the index of the ray for the current projection
% OUPUTS:
%   L - the length of the ray

% Note that since a circle is chosen as the reconstruction region, the
% length of a ray through the reconstruction region is independent of theta
r = calc_reco_radius(x, y);
if (abs(t(i)) >= r)
    L = 0;
else
    L = 2*sqrt(r^2-t(i).^2);
end

