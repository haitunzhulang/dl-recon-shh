function x = fistatv2d(F, x, data, step_size, reg_param, varargin)
% FISTATV2D Solves minimization problem using FISTA with TV regularization
% It solves the following penalized least-squares optimization problem:
%     x_hat = argmin_x F(x) + reg_param*||x||_TV
% where x is a 2-D object, F(x) is a user-supplied function and ||.||_TV 
% is the TV-norm.
% More information on the FISTA algorithm can be found in the following 
% publications:
%     Amir Beck and Marc Teboulle (2009a). A Fast Iterative Shrinkage-
%     Thresholding Algorithm for Linear Inverse Problems. SIAM J. Imag.
%     Science, 2(1): 183-202.
%
%     Amir Beck and Marc Teboulle (2009b). Fast Gradient-based Algorithms for 
%     Constrained Total Variation: Image Denosing and Deblurring Problems.
%     IEEE Trans. Image Processing, 18(11): 2419-2434.
%
%     Brenda O’Donoghue and Emmanuel Candès (2015). Adapive Restart for
%     Accelerated Gradient Schemes. Foundations of Computational
%     Mathematics, 15(3): 715-732.
% INPUTS:
%   F - function handle to a function that computes the value of the cost
%      function and its gradient, excluding the regularization term. The
%      function handle has the prototype:
%        [cost, dFdx] = F(x, data)
%      where x and data are described below, cost is scalar giving the
%      value of the cost function, and dFdx is a Nx x Ny matrix containing
%      the gradient of F with respect to x.
%   x - a Nx x Ny matrix containing an initial guess for the object to be
%       estimated
%   data - structure (or any other data type) containing any information in
%      addition to x need to compute the value of F and its gradient
%   step_size [OPT] -
%      If the function diverges, try increasing the Lipschitz constant. An
%      appropiate choice for this variable depends on the scaling of the 
%      pressure data. 
%   reg_param - the regularization parameter value. It controls the relative
%      weight between F(x, data) and the TV regularization term. Again, the
%      best choice for this value will depend on the data. 
% OPTIONS:
%   max_iter - the maximum number of iterations to perform (Default: 40)
%   min_rel_cost_diff - the function will be said to have converged if the 
%       relative difference between successive cost function values is less
%       than this amount. (Default: -inf)
%   fovinds - a two-element cell array, where fovinds{1} is the indices of rows
%       to include within the field-of-view and fovinds{2} is the same for the
%       columns. (Default: Largest square within the transducer array)
%   fovrad - radius of field-of-view (FOV) [m]. If you want a circular FOV
%       instead of rectangular, you can set this options. The option
%       fovinds must be sufficiently large to contain the entire circle.
%       (Default: inf)
%   verbose - 0, 1, or 2; the amount of information to display while the 
%       program is running; larger numbers result in more information being 
%       displayed. (Default: 1)
%   output_filename_prefix - string containing the filename prefix where 
%       intermediate estimates of the absorbed optical energy density will be
%       stored. The iteration number will be appended to the prefix. The option
%       will be ignored if verbose = 0. (Default: 'FISTA_')
% OUTPUTS:
%   x - a Nx x Ny matrix containing the estimate of the object after the 
%       final iteration.
% Examples:
%   x = fistatv2d(F, x, data, 0.1, 1E-4, 'output_filename_prefix', ...
%       'FISTA_CONSTSTEP5_TV1E-4_');
% See also: 

% Process user-supplied optional inputs
opts = process_inputs(varargin);

% If the input projection operator is a string, find the corresponding
% function.
if (ischar(opts.proj_op))
    opts.proj_op = proj_str2handle(opts.proj_op);
end

% Check compatibility of different optional inputs
if (~opts.compute_cost_x && strcmp('func', opts.adapt_restart))
    error('Function-based adapative restart requires compute_cost_x = true.');
end
 
xm1 = x; % Estimate of x at previous iteration
y = x; % Special linear combination of object estimates used in FISTA 
t = 1; % Weight used in linear combination 

if (opts.verbose >= 1)
    fprintf('%-5s%10s%20s%10s%10s%20s', 'Iter.', 'Cost_y', ...
        'Rel. Cost_y Change', '||Grad||', 'Step Size', '||x_k - x_{k-1}||');
    if (strcmpi('func', opts.compute_cost_x) || ...
            strcmpi('linesearch', opts.step_size_method))
        fprintf('%10s %20s', 'Cost_x', 'Rel. Cost_x Change');
    end
    fprintf('%15s\n', 'Duration');
    fprintf('=================================================================================================\n');
end

k = 1; % Iteration number
done = false;
cost_x = inf; % F(x_k, data)
costm1_x = inf; % F(x_{k-1}, data)
costm1_y = inf; % F(y_{k-1}, data)
while (~done)
    start_time = tic; % Start the timer
    if (opts.verbose >= 1)
        fprintf('%-5d', k);
    end
    
    % Evaluate the cost function and calculate its gradient
    [cost0_y, dFdy] = F(y, data);
    cost_y = cost0_y + reg_param*tvnorm2d(y);
    if (opts.verbose >= 1)
        fprintf('%10.5g', cost_y);
        if (k > 1)
            % Only print change in cost function value after first iteration
            fprintf('%20.5g', (costm1_y-cost_y)/cost_y);
        else
            fprintf('%20g', 0);
        end
        fprintf('%10.5g', norm(dFdy(:)));
    end

    % Display the current estimate of gradient
    if (opts.verbose >= 2)
        if (k == 1)
            hfig = figure; % Don't overwrite any existing figures
        else
            figure(hfig);
        end
        subplot(1,2,1);
        display_object(dFdy, ['Gradient, Iteration: ', num2str(k)], opts);
    end
    
    % Choose the step size either using a constant step size or a line
    % search based on user-supplied settings.
    [step_size, cost0_x, xtmp] = choose_step_size(F, y, data, step_size, ...
        reg_param, dFdy, cost0_y, opts);
    
    if (opts.verbose >= 1)
        fprintf('%10.5g', step_size);
    end
    
    % Apply the TV proximal operator (Equations 3.13/3.14 in FISTA paper)
    % This step may have already been performed as part of a line search.
    if (isempty(cost0_x))
        x = tvprox2d(y - step_size*dFdy, reg_param*step_size, opts.proj_op);
    else
        x = xtmp;
    end
    
    if (opts.verbose >= 1)
        fprintf('%20.5g', norm(x(:) - xm1(:)));
    end
    
    % Project the current estimate of x onto the TV Ball
    if (opts.tv_projection_gamma < inf)
        [x,A_,B_,C_] = perform_tv_projection(x,opts.tv_projection_gamma);
        
    end;

    % Display the current estimate of x
    if (opts.verbose >= 2)
        figure(hfig); subplot(1,2,2);
        display_object(x, ['Object, Iteration: ', num2str(k)], opts);
    end
    
    % Optionally compute the cost function at x. This is not necessary for
    % the algorithm, but may be useful for diagnostic purposes.
    if (opts.compute_cost_x || strcmpi('linesearch', opts.step_size_method))
        if (isempty(cost0_x))
            cost0_x = F(x, data);
        end
        cost_x = cost0_x + reg_param*tvnorm2d(x);
    end
    
    if ((opts.compute_cost_x || strcmpi('linesearch', opts.step_size_method)) && opts.verbose >= 1)
        fprintf('%10.5g', cost_x);
        if (k > 1)
            fprintf('%20.5g', (costm1_x - cost_x)/cost_x);
        else
            fprintf('%20g', 0);
        end
    end

    % Weight at next iteration (Equation 3.15 in FISTA paper)
    tp1 = (1+sqrt(1+4*t*t))/2;
   
    % Equation 3.16 in FISTA paper
    y = x + ((t-1)/tp1)*(x-xm1); % This does nothing if not doing TV

    % Save current state to disk
    if (~isempty(opts.output_filename_prefix))
        save([opts.output_filename_prefix, num2str(k), '.mat'], 'x', ...
            'y', 'dFdy', 't', 'cost_x', 'cost_y', 'step_size', 'reg_param');
    end
    
    t = tp1;
    
    % Optionally employ adaptive restart to accelerate convergence.
    if (~strcmpi('no', opts.adapt_restart))
        if (k > 1 && ((strcmpi('grad', opts.adapt_restart) && ...
                sum(dFdy(:) .* (x(:) - xm1(:))) > 0) || ...
            (strcmpi('func', opts.adapt_restart) && cost_x > costm1_x)))
                fprintf('restart');
                y = x;
                t = 1;
        end
    end
    
    if (opts.verbose >= 1)
        fprintf('%15.2f\n', toc(start_time));
    end

    % Check if the algorithm has converged
    done = isconverged(k, cost_y, costm1_y, x, xm1, dFdy, opts); 
    costm1_y = cost_y;
    costm1_x = cost_x;
    xm1 = x;
    k = k + 1;
end

% ========================================================================
function opts = process_inputs(inputs)

ALLOWED_TV_TYPE = {'aniso', 'iso'};
ALLOWED_ADAPT_RESTART = {'no', 'grad', 'func'};
ALLOWED_STEP_SIZE_METHOD = {'const', 'linesearch'};
ALLOWED_INIT_STEP_SIZE_METHOD = {'const', 'bb'};

check_fovinds = @(x)(iscell(x) && numel(x) == 2 && isvector(x{1}) && ...
    isvector(x{2}) && all(floor(x{1}) == x{1}) && all(floor(x{2}) == x{2}) ...
    && all(x{1} > 0) && all(x{2} > 0));
isnonnegnum = @(x)(isscalar(x) && x >= 0);
isnonnegint = @(x)(isnonnegnum(x) && floor(x) == x);
isposnum = @(x)(isscalar(x) && x > 0);
isposint = @(x)(isposnum(x) && floor(x) == x);

p = inputParser;
p.addParameter('tvtype', 'aniso', @(x) any(validateString(x, ALLOWED_TV_TYPE)));
% ========================================================================
% Stopping criteria settings
p.addParameter('max_iter', 40, isposint);
p.addParameter('min_rel_cost_diff', -inf, @(x)(isnumeric(x) && numel(x) == 1));
p.addParameter('min_norm_x_diff', 0, isnonnegnum);
p.addParameter('min_norm_grad', 0, isnonnegnum);
% ========================================================================
% Adding in TV Projection
p.addParameter('tv_projection_gamma',inf,isnonnegnum);
% ========================================================================
p.addParameter('fovinds', {}, check_fovinds); % Set default value later
p.addParameter('fovrad', inf, isposnum);
p.addParameter('verbose', 1, isnonnegint);
p.addParameter('output_filename_prefix', 'FISTA_', @ischar);
p.addParameter('proj_op', @(x)(x), @(x)(isa(x, 'function_handle') || ischar(x)));
p.addParameter('adapt_restart', 'no', ...
    @(x) any(validatestring(x, ALLOWED_ADAPT_RESTART)));
p.addParameter('step_size_method', 'const', ...
    @(x) any(validatestring(x, ALLOWED_STEP_SIZE_METHOD)));
p.addParameter('init_step_size_method', 'const', ...
    @(x) any(validatestring(x, ALLOWED_INIT_STEP_SIZE_METHOD)));
p.addParameter('compute_cost_x', false, @islogical);
p.addParameter('eta', 0.5, @(x) (isscalar(x) && x > 0 && x < 1));

parse(p, inputs{:});
opts = p.Results;

% ========================================================================
function [step_size, cost0_x, xtmp] = choose_step_size(F, y, data, ...
    step_size, reg_param, dFdy, cost0_y, opts)

if (strcmp('const', opts.step_size_method))
    cost0_x = [];
    xtmp = [];
    return;
end

% ========================================================================
% TODO: Exploit the fact that the forward operator is linear to reduce the
% number of times F(...) is evaluated, i.e. 
%    H(x + step_size*dFdx) = Hx + step_size H dFdx
% if H is linear and dFdx is in the domain of H
% ========================================================================
% Use line search to choose step size. See Beck and Teboulle 2009a p. 194.
xtmp = tvprox2d(y - step_size*dFdy, reg_param*step_size, opts.proj_op);
cost0_x = F(xtmp, data);
while (cost0_x - cost0_y > sum( (xtmp(:) - y(:)).*dFdy(:) ) + ...
        0.5/step_size*norm(xtmp(:) - y(:))^2)
    fprintf('%10.3g%10.5g\n', step_size, cost0_x);
    step_size = step_size * opts.eta;
    xtmp = tvprox2d(y - step_size*dFdy, reg_param*step_size, opts.proj_op);
    cost0_x = F(xtmp, data);
end

% ========================================================================
function display_object(x, titlestr, opts)

% ix = opts.fovinds{1}; iy = opts.fovinds{2};
% imagesc(1E3*kgrid.x_vec(ix), 1E3*kgrid.y_vec(iy), x(ix,iy)); 
% xlabel('mm', 'fontsize', 18);
% ylabel('mm', 'fontsize', 18);
imagesc(x);
axis image; colorbar;
set(gca, 'fontsize', 18);        
title(titlestr, 'fontsize', 18);
drawnow;

% ========================================================================
function bool = isconverged(k, cost, costm1, x, xm1, dFdy, opts)

bool = (k >= opts.max_iter || ...
    (k > 1 && (costm1-cost)/cost < opts.min_rel_cost_diff) || ...
    (k > 1 && norm(x(:) - xm1(:)) < opts.min_norm_x_diff) || ...
    norm(dFdy(:)) < opts.min_norm_grad);

% ========================================================================
