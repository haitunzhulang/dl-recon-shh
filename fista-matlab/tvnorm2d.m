function val = tvnorm2d(x, tvtype)
% TVNORM2D Computes the total-variation (TV) semi-norm of a 2-D object
% INPUTS:
%   x - a Nx x Ny matrix
%   tvtype [OPT] - either 'iso' or 'aniso', the type of the discrete TV 
%     implementation. For the 'iso', short for isotropic, the TV semi-norm
%     is given by
%       ||x||_{TV} = \sum_i \sum_j \sqrt{ (x_{i,j} - x_{i+1,j})^2 + 
%           (y_{i,j} - y_{i+1,j})^2 }
%     For the 'aniso', short for anisotropic, the TV semi-norm is given by
%       ||x||_{TV} = \sum_i \sum_j { |x_{i,j} - x_{i,j+1}| + 
%           |x_{i,j} - x_{i+1,j}| }
%     This is the terminology employed by Beck and Teboulle, IEEE Trans. on 
%     Image Processing, 2009 (pg. 4). [Default: 'iso']
% OUTPUTS:
%   val - a scalar value equal to the TV semi-norm
% Examples:
%   cost = tvnorm2d(x);
%   cost = tvnorm2d(x, 'aniso');
% See also:

% Set default value for TV norm type
if (~exist('tvtype', 'var') || isempty(tvtype))
    tvtype = 'aniso';
end

% Check that the object is in the expected format
if (~isnumeric(x))
    error('The object should be a numeric matrix.');
end
if (~ismatrix(x))
    error('The object should be 2-D.');
end


% [Nx, Ny] = size(x);
% 
% dx = zeros(size(x));
% dy = zeros(size(x));
% 
% dx(1:(Nx-1),:) = x(1:(Nx-1),:) - x(2:Nx,:);
% dy(:,1:(Ny-1)) = x(:,1:(Ny-1)) - x(:,2:Ny);

dx = diff(x, 1, 1);
dy = diff(x, 1, 2);

% size(dx)
% size(dx2)
% 
% all(all(eq(dx, dx2)))
% all(all(eq(dy, dy2)))

if (strcmp('iso', tvtype))
    val = sum(sqrt(dx(:).*dx(:) + dy(:).*dy(:))); 
else
    val = sum(abs(dx(:)) + abs(dy(:)));
end

