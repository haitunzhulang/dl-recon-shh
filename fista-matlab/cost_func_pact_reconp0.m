function [cost, dFdx] = cost_func_pact_reconp0(x, data)
% COST_FUNC_PACT_RECONP0 Cost function for 

psim = forward_prop(data.kgrid, data.medium, data.sensor, x, ...
    data.kwave_options);

cost = norm(data.pmeas(:)-psim(:))^2;

if (nargout > 1)
    pdiff = psim - data.pmeas;
    dFdx = backward_prop(data.kgrid, data.medium, data.sensor, pdiff, ...
       data.kwave_options);    
    dFdx(sqrt(data.kgrid.x.^2 + data.kgrid.y.^2) > data.fovrad) = 0;
    dFdx(data.medium.sound_speed < 500 | data.medium.density < 500) = 0;
end

% =============================================================================
function p = forward_prop(kgrid, medium, sensor, A, kwave_options)
source.p0 = A;
% p = kspaceFirstOrder2D(kgrid, medium, source, sensor, kwave_options{:});
% Use of evalc suppresses Command Window output
evalc('p = kspaceFirstOrder2D(kgrid, medium, source, sensor, kwave_options{:});');
% ============================================================================
function A = backward_prop(kgrid, medium, sensor, pdiff, kwave_options)

source.p0 = 0;
sensor.adjoint_data = pdiff;
% A = kspaceFirstOrder2D_adjoint(kgrid, medium, source, sensor, ...
%     kwave_options{:});
% Use of evalc suppresses Command Window output
evalc(['A = kspaceFirstOrder2D_adjoint(kgrid, medium, source, sensor,', ...
    'kwave_options{:});']);
