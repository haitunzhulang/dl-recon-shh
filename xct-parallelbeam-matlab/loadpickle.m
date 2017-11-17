function [a] =  loadpickle(filename)
% LOADPICKLE ... 
%  
%  https://xcorr.net/2013/06/12/load-pickle-files-in-matlab/

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 29-Jun-2017 15:15:02 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : loadpickle.m 


tempname = '/home/shenghua/DL-recon/xct-parallelbeam-matlab/tmp/tmp_1';

if ~exist(filename,'file')
    error('%s is not a file',filename);
end
outname = [tempname '.mat'];
pyscript = ['import _pickle as pickle;import sys;import scipy.io;file=open("' filename '","rb");dat=pickle.load(file);file.close();scipy.io.savemat("' outname '",dat)'];
system(['python3 -c ''' pyscript '''']);
a = load(outname);


%LD_LIBRARY_PATH=/opt/intel/composer_xe_2013/mkl/lib/intel64:/opt/intel/composer_xe_2013/lib/intel64;





 
% ===== EOF ====== [loadpickle.m] ======  
