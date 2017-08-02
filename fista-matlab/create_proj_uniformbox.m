function proj = create_proj_uniformbox(a, b)
% CREATE_PROJ_UNIFORMBOX Creates function handle to projection operator 
% where the projection operator maps each element of x to be between a and 
% b inclusive.

function x = fproj(x)

x(x<a) = a;
x(x>b) = b;

end


proj = @fproj;

end

