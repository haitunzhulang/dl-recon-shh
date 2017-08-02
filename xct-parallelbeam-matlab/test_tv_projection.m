function test_tv_projection()
% TEST_TV_PROJECTION ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 27-Jun-2017 15:54:56 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : test_tv_projection.m 


dim = 25;
img = zeros(dim,dim);
for i=1:dim
    for j=1:dim
        img(i,j) = (i-1)*dim+(j-1);
    end;
end;

% Check that compute_total_variation gives the same -- it does
tv = compute_total_variation(img);

% Check that div gives the same - done
t = grad(img);
t = -div( t );

% Check perform_vf_normalization
t = grad(img);
t = my_perform_vf_normalization(t);

% Check to see if it will work:
options.niter = 1000;
options.verbose = 1;
options.xtgt = img;
[x,err_tv,err_l2, err_tgt] = perform_tv_projection(img,13000,options);
 
% ===== EOF ====== [test_tv_projection.m] ======  
