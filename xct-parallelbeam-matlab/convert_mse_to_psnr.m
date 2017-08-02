function [psnr] = convert_mse_to_psnr(mse)
% CONVERT_MSE_TO_PSNR ... 
%  This is hacky, only works for matrices, and assumes the maximum range is
%  1.
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 29-Mar-2017 13:16:39 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : convert_mse_to_psnr.m 


psnr = 10*log10(1./(mse));





% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [convert_mse_to_psnr.m] ======  
