function [cost, dFdx] = cost_func_pact_temp(x, data)
% COST_FUNC_PACT_TEMP Cost function for estimating temperature

p0diff = data.p0heated - data.a0*data.Aref - data.a1*x.*data.Aref;
Tdiff = x - data.Tref;

% DxT = diff(x, 1, 1);
% DyT = diff(x, 1, 2);
% 
% DxT = [zeros(1, size(x,2)); DxT];
% DyT = [zeros(size(x,1),1) DyT];

cost = 0.5*sum(p0diff(:).^2) + 0.5*data.lambda1*sum(Tdiff(:).^2);% + ...
%     0.5*data.lambda2*(sum(DxT(:).^2) + sum(DyT(:).^2));

dFdx = -1*data.a1*data.Aref .* p0diff + data.lambda1*Tdiff;% + ...
%     data.lambda2*(DxT + DyT);
