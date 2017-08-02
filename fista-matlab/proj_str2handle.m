function hproj = proj_str2handle(projstr)
% PROJ_STR2HANDLE Converts projection operator name to function handle
% INPUTS:
%   projstr - a string containing the name of a projection operator; must
%     be one of the following:
%       'none': projects onto all reals
%       'nonneg': projects onto the non-negative reals
%       'uniformbox#a,#b': projects each element onto the interval [#a, #b]
%          where #a is a number indicating the start of the interval and #b
%          is a number indicating the end of the interval
% OUTPUTS:
%   hproj - a function handle to a projection operator which takes a single
%       input and returns a single output
% Examples:
%   hproj = proj_str2handle('none');
%   hproj = proj_str2handle('pos');
%   hproj = proj_str2handle('uniformbox1.3,1.7');
% See also: proj_positive, create_proj_uniformbox

nbox = length('uniformbox');
if (strcmpi('none', projstr))
    hproj = @(x)(x);
elseif (strcmpi('nonneg', projstr))
    hproj = @proj_nonneg;
elseif (nbox < length(projstr) && ...
        strcmpi('uniformbox', projstr(1:nbox)))
    MATCH_STR = 'uniformbox([\-\+eE\d.]+),([\-\+eE\d.]+)';
    matches = regexpi(projstr, MATCH_STR, 'tokens', 'once');
    if (length(matches) == 2)
        hproj = create_proj_uniformbox(str2double(matches{1}), ...
            str2double(matches{2}));
    else
        error('Unrecognized format for uniform box projection operator.');
    end
else
    error('Unrecognized projection operator.');
end

