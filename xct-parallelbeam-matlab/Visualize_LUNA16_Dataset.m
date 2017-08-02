function Visualize_LUNA16_Dataset()
% VISUALIZE_LUNA16_DATASET ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 26-May-2017 13:44:03 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : Visualize_LUNA16_Dataset.m 


mf = '/home/bmkelly/medception/LUNG_Time/matlab_format/';
files = dir([mf '*.mat']);

%%
for i=50:150
    %
    for j=0:3
        
        f = dir([mf num2str(i) '_' num2str(j) '.mat']);
        if length(f) > 0
            disp(['Visualizing for ' num2str(i) '_' num2str(j)]);
            
            load([mf f.name]);
            sz = size(ArrayDicom);
            figure(1);
            c_min = min(ArrayDicom(:)); c_max = max(ArrayDicom(:));
            for k=1:sz(3)
                imagesc(squeeze(ArrayDicom(:,:,k)));
                caxis([c_min c_max]);
%                 colorbar;
                title(['Slice ' num2str(k) '/' num2str(sz(3)) ' for ' num2str(i) '-' num2str(j)]);
                drawnow();
                pause(.01);
            end;
        end;
    end;
end;




 
% ===== EOF ====== [Visualize_LUNA16_Dataset.m] ======  
