function create_images_measured_data_for_all()
% CREATE_IMAGES_MEASURED_DATA_FOR_ALL ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 30-May-2017 16:46:11 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : create_images_measured_data_for_all.m 

mf ='images_and_measured_data_for_all/';
num_images = 10000;

MIN_ELLIPSES=3;
MAX_NELLIPSES=9;
theta = (-90:89)*pi/180;
for i=1:num_images
    num_ellipses = round((MAX_NELLIPSES-MIN_ELLIPSES)*rand() + MIN_ELLIPSES);
    
    [img,g,ellipses] = simulate_projection(NX, theta,num_ellipses);
    fname = [mf num2str(i) '.mat'];
    
    save(fname,'img','g','ellipses');
end;






 
% ===== EOF ====== [create_images_measured_data_for_all.m] ======  
