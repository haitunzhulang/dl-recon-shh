function proj = Projection(theta,centerx,centery,major,minor,rangle,rho,len_xp,t_shift)
% PROJECTION ... 
%  via elipse parameters
%  
 
%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 03-Jan-2017 11:07:13 
%% Revision : 1.00 
%% Developed : 8.4.0.150421 (R2014b) 
%% Filename  : Projection.m 
 
if nargin < 9
    len_xp = 256;
    t_shift = 0;
    disp('Using default params for len_xp and t_shift');
end;
% t_lim = [-4,4];
% rho = 1;
% keyboard();
t = linspace(-2,2,len_xp)'-t_shift;
 
a2 = calc_a_squared(major,theta,minor);
 
relevant_t = abs(t) <= sqrt(a2);
 
if rangle ==0 && centerx ==0 && centery == 0
%     keyboard();
    proj = relevant_t.*(((2*rho*major*minor)/a2)*sqrt(a2-t.^2.*relevant_t));
    return;
end;
if isnan(centery/centerx)
    epsilon = .0001;
    centerx = centerx+epsilon;
%     centery = centery+epsilon;
    t_shift = sqrt(centerx^2+centery^2)*cosd(atand(centery/centerx)-theta);
else
   t_shift = sqrt(centerx^2+centery^2)*cosd(atand(centery/centerx)-theta);
end
 
% This shouldn't have to be here.  Why does this fix it?...
% Solved: I should have been using atan2
if centerx <0 
    t_shift = -1*t_shift;
end;
proj = Projection(theta-rangle,0,0,major,minor,0,rho,len_xp,t_shift);
 
 
 
 
 
 
end
 
function asquared = calc_a_squared(A,theta,B)
asquared = A^2*cosd(theta)^2 + B^2*sind(theta)^2;
end
 
 
 
 
 
 
 
 
% ===== EOF ====== [Projection.m] ======  



