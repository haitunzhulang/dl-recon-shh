function write_system_matrix(H, fn_prefix)
% WRITE_SYSTEM_MATRIX Write sparse system matrix to disk as three arrays
% INPUTS:
%   H - m x n sparse matrix
%   fn_prefix - filename prefix
% OUTPUTS:
%   Three single-precision arrays will be written to disk with the indices
%   of the rows, the indices of the cols, and the values of the non-zero
%   elements of the system matrix. The files will be written to the current
%   directory with names of the form:
%      (1) <fn_prefix>_irows.dat
%      (2) <fn_prefix>_icols.dat
%      (3) <fn_prefix>_vals.dat
% Examples:
%   write_system_matrix(H120v3, 'H120v3');
% See also:

% Find indices and values of non-zeros elements of H
%  keyboard();
[ix, iy] = find(H);
vals = nonzeros(H);

fid = fopen([fn_prefix, '_irows.dat'], 'wb');
fwrite(fid, ix, 'float');
fclose(fid);

fid = fopen([fn_prefix, '_icols.dat'], 'wb');
fwrite(fid, iy, 'float');
fclose(fid);

fid = fopen([fn_prefix, '_vals.dat'], 'wb');
fwrite(fid, vals, 'float');
fclose(fid);
