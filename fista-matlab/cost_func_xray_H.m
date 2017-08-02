function [cost, dFdx] = cost_func_xray_H(x, data)
% COST_FUNC_XRAY_H Cost function for X-ray absorption for sparse matrix-based formulation of forward operator

[nx, ny] = size(x);

res = data.H*x(:) - data.g;

cost = 0.5*sum(res(:).^2);

% Pre-compute the transpose of H
dFdx = data.H'*res;

dFdx = reshape(dFdx, nx, ny);


