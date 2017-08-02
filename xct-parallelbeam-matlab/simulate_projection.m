function [im,g,ellipses] = simulate_projection(NX, theta,num_ellipses,ellipses)
% SIMULATE_PROJECTION ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 18-Apr-2017 12:03:18 
%% Revision : 1.00 
%% Developed : 9.0.0.341360 (R2016a) 
%% Filename  : simulate_projection.m 

% NX = 256;
% theta = (-70:69)*pi/180;
% 
% num_ellipses =6;
% rng(1337);
if ~exist('ellipses')

    ellipses = rand(num_ellipses, 6);

    %Starting ellipse:
    MIN_CENTER = -0.01;
    MAX_CENTER = 0.01;
    MIN_AXES_LEN = 0.5; 
    MAX_AXES_LEN = 0.7;
    MIN_ROT_ANGLE = -.1*pi;
    MAX_ROT_ANGLE = .1*pi;
    MIN_INTENS = 0.2;
    MAX_INTENS = .7;

    ellipses(1,1:2) =(MAX_CENTER - MIN_CENTER)*ellipses(1,1:2) + ...
        MIN_CENTER;
    ellipses(1,3:4) = (MAX_AXES_LEN - MIN_AXES_LEN)*ellipses(1,3:4) + ...
        MIN_AXES_LEN;
    ellipses(1,5) = (MAX_ROT_ANGLE - MIN_ROT_ANGLE)*ellipses(1,5) + ...
        MIN_ROT_ANGLE;
    ellipses(1,6) = (MAX_INTENS - MIN_INTENS)*ellipses(1,6) + MIN_INTENS;

    im_main = sim_image(NX, ellipses(1,:),0);

    % Rest of ellipses
    MIN_CENTER = -0.3;
    MAX_CENTER = 0.3;
    MIN_AXES_LEN = 0.05; 
    MAX_AXES_LEN = 0.4;
    MIN_ROT_ANGLE = 0;
    MAX_ROT_ANGLE = 2*pi;
    MIN_INTENS = .1;
    MAX_INTENS = .5;


    constraints_satisfied = 0;

    for count=2:num_ellipses
        constraints_satisfied=0;
        while ~(constraints_satisfied)
            constraints_satisfied = 1;
            ellipses(count,:) = rand(1,6);
            ellipses(count,1:2) =(MAX_CENTER - MIN_CENTER)*ellipses(count,1:2) + ...
                MIN_CENTER;
            ellipses(count,3:4) = (MAX_AXES_LEN - MIN_AXES_LEN)*ellipses(count,3:4) + ...
                MIN_AXES_LEN;
            ellipses(count,5) = (MAX_ROT_ANGLE - MIN_ROT_ANGLE)*ellipses(count,5) + ...
                MIN_ROT_ANGLE;
            ellipses(count,6) = (MAX_INTENS - MIN_INTENS)*ellipses(count,6) + MIN_INTENS;
            if rand(1) > .5
                ellipses(count,6) = -1*ellipses(count,6);
            end;


            im_current = sim_image(NX,ellipses(1:count,:),0);

        %     subplot(2,2,1);
        %     imagesc(im_main);
        %     colorbar;
        %     subplot(2,2,2);
        %     imagesc(im_current);
        %     colorbar;
        %     sum(im_current(:) < 0)
        %     ff = sum(im_main(:) >0) ~= sum(im_current(:)>0)
        %     pause();


            % Cannot be < 0
            if sum(im_current(:) < 0)>0
                constraints_satisfied = 0;
            end;

            % Cannot have ellipse out of main ellipse
            if sum(im_main(:) >0) ~= sum(im_current(:)>0)
                constraints_satisfied = 0;
            end;
        end;
    end;
else
    sz = size(ellipses);
    num_ellipses = sz(1);
end;


theta = theta*180/pi;
proj = zeros([NX length(theta) num_ellipses]);
for ellipse=1:num_ellipses
    for i=1:length(theta)
        proj(:,i,ellipse) = Projection(theta(i),ellipses(ellipse,1)*2,ellipses(ellipse,2)*2, ...
            ellipses(ellipse,3)*2,ellipses(ellipse,4)*2,ellipses(ellipse,5)*360/(2*pi),ellipses(ellipse,6)/2,NX,0);%*(NX/4),NX,0);
    end;
end;


% ellipses(:,5) = ellipses(:,5)/180;

im = sim_image(NX, ellipses,0);

squash_proj = sum(proj,3);
g=squash_proj;

% Smoothing
SMOOTH_FILTER_SIZE = [9, 9];

SMOOTH_FILTER_WIDTH = .75;

h = fspecial('gaussian', SMOOTH_FILTER_SIZE, SMOOTH_FILTER_WIDTH);
h_meas = fspecial('gaussian', [SMOOTH_FILTER_SIZE(1) 1 ], SMOOTH_FILTER_WIDTH);

g_preblur =g;
im_preblur=im;

im =  imfilter(im, h, 'replicate');
g = imfilter(g, h_meas, 'replicate');

drawing=0;
if drawing
    
    
    
    subplot(2,3,1);
    imagesc(g_preblur);
    title('Pre blur measured data');
    subplot(2,3,3);
    imagesc(g);
    title('Post blur Measured Data');
    subplot(2,3,2);
    imagesc(g-g_preblur);
    title('Difference in measured data');
%     subplot(2,3,5);
%     imagesc(iradon(g,theta));
%     title('Iradon reconstruction');
%     colorbar;
    subplot(2,3,4);
    imagesc(im);
    title('Original Image');
%     colorbar;
end;



% ratio = max(max(im))/(max(max(iradon(squash_proj,theta))));
% ratio2 = NX/ratio
% ratio3 = NX^2/ratio

% ===== EOF ====== [simulate_projection.m] ======  
