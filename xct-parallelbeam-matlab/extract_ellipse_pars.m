function [x0, y0, A, B, alpha, rho] = extract_ellipse_pars(ellipses, i)
% EXTRACT_ELLIPSE_PARS Pulls out ellipse parameters for readability

x0 = ellipses(i,1);
y0 = ellipses(i,2);
A = ellipses(i,3);
B = ellipses(i,4);
alpha = ellipses(i,5);
rho = ellipses(i,6);