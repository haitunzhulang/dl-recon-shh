function y = tvprox2d(x, lambda, proj_op, varargin)
% TVPROX2D Solves total-variation (TV) proximal problem for a 2-D object
% A. Beck and M. Teboulle, Fast gradient-based algorithms for constrained
% total variation image denoising and deblurring problems, IEEE Trans. on
% Image Processing, 18(11): 2419-2434, 2009.
% Algorithm FGP (Eqns. 4.9 - 4. 11)

% Set default values for optional inputs
if (~exist('proj_op', 'var') || isempty(proj_op))
    proj_op = @(x)(x); % Project onto all reals
end

% If the projection operator is a string, assume it represents the name of
% the projection operator, and find the corresponding function handle.
if (ischar(proj_op))
    proj_op = proj_str2handle(proj_op);
end

% If lambda is approximately equal to zero, the solution of the proximal
% problem is given by the application of the projection operator.
if (lambda <= eps)
    y = proj_op(x);
    return;
end

opts = process_inputs(varargin);

[Nx, Ny] = size(x);

p = zeros(Nx-1, Ny);
q = zeros(Nx, Ny-1);
pm1 = p;
qm1 = q;
r = p;
s = q;
t = 1;

if (opts.verbose)
    fprintf('%-5s %10s %20s\n', 'Iter.', 'Cost', 'Rel. Cost Change');
    fprintf('=======================================\n');
end

k = 1;
done = false;
cost = inf;
costm1 = inf;
% Solve the TV proximal problem using the fast gradient projection method
while (~done)
    % Equation 4.9
    [rtmp, stmp] = imneggrad2d(proj_op(x - lambda*imdiv2d(r, s)));
    [p, q] = proj_p(r + rtmp/(8*lambda), s + stmp/(8*lambda), opts.tvtype);
    
    if (opts.verbose)
        y = proj_op(x - lambda*imdiv2d(p, q));
        cost = 0.5*sum( (y(:) - x(:)).^2 ) + ...
           lambda*tvnorm2d(y, opts.tvtype);
        fprintf('%-5d %10.5g', k, cost);
        if (k > 1)
            fprintf(' %20.5g\n', (costm1 - cost)/cost);
        else
            fprintf('\n');
        end
    end
  
    % Equation 4.10
    tp1 = (1+sqrt(1+4*t*t))/2;
  
    % Equation 4.11
    r = p + ((t-1)/tp1)*(p - pm1);
    s = q + ((t-1)/tp1)*(q - qm1);
  
    pm1 = p;
    qm1 = q;
    t = tp1;
    done = isconverged(k, cost, costm1, opts);
    costm1 = cost;
    k = k + 1;
end
y = proj_op(x - lambda*imdiv2d(p, q));


% ========================================================================
function opts = process_inputs(inputs)

isposnum = @(x)(isnumeric(x) && numel(x) == 1 && x > 0);
isposint = @(x)(isposnum(x) && floor(x) == x);

p = inputParser;
p.addParameter('max_iter', 100, isposint);
p.addParameter('min_rel_cost_diff', -inf, @isscalar);
p.addParameter('tvtype', 'aniso');
p.addParameter('verbose', false, @islogical);

parse(p, inputs{:});

opts = p.Results;
% ========================================================================
function A = imdiv2d(X, Y)

Nx = size(Y,1);
Ny = size(X,2);

% Ensure that output has dimensions Nx x Ny
X1 = zeros(Nx+1,Ny);
X1(2:Nx,:) = X;
Y1 = zeros(Nx,Ny+1);
Y1(:,2:Ny) = Y;

%A = X1(2:Nx+1,:) - X1(1:Nx,:) + Y1(:,2:Ny+1) - Y1(:,1:Ny);
A = diff(X1, 1, 1) + diff(Y1, 1, 2);
% ========================================================================
function [X, Y] = imneggrad2d(x)

% [Nx, Ny] = size(x);
% 
% X = x(1:Nx-1,:) - x(2:Nx,:);
% Y = x(:,1:Ny-1) - x(:,2:Ny);

X = -1*diff(x, 1, 1);
Y = -1*diff(x, 1, 2);

% ========================================================================
function [X1, Y1] = proj_p(X, Y, tvtype)

if (~exist('tvtype', 'var') || isempty(tvtype))
    tvtype = 'aniso';
end

Nx = size(Y,1);
Ny = size(X,2);

X1=zeros(Nx-1,Ny);
Y1=zeros(Nx,Ny-1);

if (strcmp('aniso', tvtype))
    X1 = X./max(1, abs(X));
    Y1 = Y./max(1, abs(Y));
else
    denom = sqrt( X(1:Nx-1,1:Ny-1).*X(1:Nx-1,1:Ny-1) + ...
             Y(1:Nx-1,1:Ny-1).*Y(1:Nx-1,1:Ny-1) );
    denom(denom<1) = 1;

    X1(1:Nx-1,1:Ny-1) = X(1:Nx-1,1:Ny-1)./denom;
    Y1(1:Nx-1,1:Ny-1) = Y(1:Nx-1,1:Ny-1)./denom;

%     denom = abs(X(:,Ny));
%     denom(denom < 1) = 1;
%     X1(:,Ny) = X(:,Ny)./denom;
    X1(:,Ny) = X(:,Ny)./max(1, abs(X(:,Ny)));

%     denom = abs(Y(Nx,:));
%     denom(denom < 1) = 1;
%     Y1(Nx,:)=Y(Nx,:)./denom;
    Y1(Nx,:) = Y(Nx,:)./max(1, abs(Y(Nx,:)));
end
% ========================================================================
function bool = isconverged(k, cost, costm1, opts)

bool = (k >= opts.max_iter || ...
    (k > 1 && (costm1-cost)/cost < opts.min_rel_cost_diff));

% ========================================================================
