function w = calc_weights_binary(theta, x, y, t, i)
% CALC_WEIGHTS_BINARY Calculates binary weights for each image pixel
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

j = mod(i-1, length(t))+1; % Find the position in t for the current ray
k = ceil(i/length(t)); % Find the angle of the current ray
dt = t(2) - t(1);
tm = x*cos(theta(k)) + y*sin(theta(k));
inds = (tm >= t(j)-dt/2 & tm < t(j)+dt/2);
w = inds(:);