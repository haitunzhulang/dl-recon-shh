function x = proj_nonneg(x)
% PROJ_NONNEG Projection operator that projects onto non-negative reals

x(x<0) = 0;

