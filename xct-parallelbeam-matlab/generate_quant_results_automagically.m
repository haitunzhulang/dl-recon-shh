function generate_quant_results_automagically()
% GENERATE_QUANT_RESULTS_AUTOMAGICALLY ... 
%  
%  1.  Showcase the loss plots from the multiple runs.
%  2.  Showcase some result images from the runs -- images at each
%  projection
%  3.  Showcased overall accuracy for different stopping criteria (1-15
%  projections, when the projections stop changing meaningfully.
%

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 29-Jun-2017 14:55:58 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : generate_quant_results_automagically.m 

%% Run information
% name_of_exp = 'Experiment_7_12_';
% name_of_exp = 'Experiment_3_0_';
% name_of_exp = 'Experiment_2_6_';
% name_of_exp = 'Experiment_4_24_';
name_of_exp = 'Experiment_4_64_';
%% 1. Showcase loss plots from each of the available stageX_model folders
% name_of_exp = 'Experiment_4_12_';
% name_of_exp = 'Experiment_4_24_';
available_model_folders = dir(['../dl-limitedview-prior/training_runs/' name_of_exp '/*_model']);

close all;
for i=1:length(available_model_folders)
    training_loss_file = [available_model_folders(i).folder '/' available_model_folders(i).name '/training_nums.out'];
    if exist(training_loss_file,'file')>0
        M = csvread(training_loss_file);
        figure;
        clf;
        hold on;
        plot(M(1,:),'r-');
        plot(M(2,:),'b-');
        xlabel('Epoch');
        ylabel('Loss');
        legend('Training Loss','Validation Loss');
        title(strrep([name_of_exp ' ' available_model_folders(i).name],'_',' '));
    else
        disp(['Training loss file does not exist:' training_loss_file]);
    end;
end;

%% 2.  Showcase result images from different stages -- need to pull this out of datasets
index = 77; % validation image

stage=2;
pickle_file = ['../dl-limitedview-prior/datasets/v' ...
    name_of_exp num2str(stage) '_AD' ...
    '/' num2str(index) '.pkl' ];
figure(1);
clf;
[ha,pos] = tight_subplot(4,8,[.02 .01],[.01 .03]);
set(gcf,'Position',[19 151 1664 796]);

if exist(pickle_file,'file')>0
    M = loadpickle(pickle_file);
    if size(M.X_train,1) < 320
        error(['file:datasets/v' name_of_exp num2str(stage) '_AD/' num2str(index) '.pkl did not have 320 items.  it had: ' num2str(size(M,1))]);
    end;
    for i=1:32
        axes(ha(i));
        imagesc(squeeze(M.X_train(i,:,:)));
        axis off;
    end;
    suptitle(['Before and after each of 16 projections, for ' name_of_exp ' stage' num2str(stage)]);
else
    disp(['File does not exist:' pickle_file]);
end;


%% 3.1 Showing some visuals of images and their corresponding MSE plots  
start_index = 76; % validation image
end_index = 99;

results_before = zeros([1 16]);
results_after = zeros([1 16]);

figure(6);
clf;
set(gcf,'Position',[1297 222 436 737]);

figure(7);
clf;
[ha1,pos] = tight_subplot(4,8,[.03 .005],[.01 .01],[.001 .001]);
set(gcf,'Position',[2 223 1293 736]);


for index=start_index:end_index
    stage=2;
    pickle_file = ['../dl-limitedview-prior/datasets/v' ...
        name_of_exp num2str(2) '_AD' ...
        '/' num2str(index) '.pkl' ];
    if exist(pickle_file,'file')>0
        M = loadpickle(pickle_file);
        if size(M.X_train,1) < 320
            error(['file:datasets/v' name_of_exp num2str(stage) '_AD/' num2str(index) '.pkl did not have 320 items.  it had: ' num2str(size(M,1))]);
        end;
        
        mean_squared_error = mean(mean((M.X_train-M.y_train).^2,2),3);
        size(mean_squared_error)
        for i=1:10
            results_before(1+(start_index-index)*10+(i-1),:) = mean_squared_error(1+32*(i-1):2:1+32*i-2);
            results_after(1+(start_index-index)*10+(i-1),:) = mean_squared_error(2+32*(i-1):2:2+32*i-2);
            size(results_before)
                        
            figure(7);
            for j=1:32
                axes(ha1(j));
                imagesc(squeeze(M.X_train((i-1)*32+j,:,:)));
                axis off;
            end;
            
            figure(6);
            clf;
            [ha,pos] = tight_subplot(2,1,[.1 .05],[.01 .05],[.3 .05]);
            axes(ha(1));
            hold on;
            plot(results_before(1+(start_index-index)*10+(i-1),:),'ro-');
            hold on;
            plot(results_after(1+(start_index-index)*10+(i-1),:),'bo-');
            xlabel('Each of the 16 projections');
            ylabel('MSE');
            legend('Before projection','After projection');
            title(['Validation example:' num2str(index*10 + i)]); 
%             axis on;
            
            axes(ha(2));
            imagesc(squeeze(M.y_train((i-1)*32+1,:,:)));
            title('Target Image');
            axis off;
            
            disp(['Just plotted for ' num2str(index*10 + i)]);
            drawnow();
            pause();
        end;
    end;
    
    
end; 


%% 3.2  Calculating the MSE for different projection numbers
name_of_exp = 'Experiment_4_64_';
available_model_folders = dir(['../dl-limitedview-prior/training_runs/' name_of_exp '/*_model']);
start_index = 75; % validation image
end_index = 100;
stage=2;

results_before = zeros([1 16 length(available_model_folders)]);
results_after = zeros([1 16 length(available_model_folders)]);

for model=0:length(available_model_folders)-1
    if model==1
        continue
    end
 
    for index=start_index:end_index

        tic;
        pickle_file = ['../dl-limitedview-prior/datasets/v' ...
            name_of_exp num2str(model) '_AD' ...
            '/' num2str(index) '.pkl' ];
        if exist(pickle_file,'file')>0
            M = loadpickle(pickle_file);
            if size(M.X_train,1) < 320
                error(['file:datasets/v' name_of_exp num2str(stage) '_AD/' num2str(index) '.pkl did not have 320 items']);
            end;

            mean_squared_error = mean(mean((M.X_train-M.y_train).^2,2),3);
            for i=1:10
                results_before(1+(index-start_index)*10+(i-1),:,model+1) = mean_squared_error(1+32*(i-1):2:1+32*i-2);
                results_after(1+(index-start_index)*10+(i-1),:,model+1) = mean_squared_error(2+32*(i-1):2:2+32*i-2);
            end;
        else
            disp([pickle_file ' did not exist']);
        end;
        disp(['Finished with ' num2str(index) ' for model: ' num2str(model)]);
        toc;
    end;
end;

results_before_total = results_before;
results_after_total = results_after;

%% 3.2 Continute: Calculate statistics given results_before/after
name_of_exp = 'Experiment_4_64_';
figure(5);
clf;
for i=1:length(available_model_folders)
    subplot(1,length(available_model_folders),i);
    hold on;
    plot(mean(squeeze(results_before_total(:,:,i)),1),'ro-');
    plot(mean(squeeze(results_after_total(:,:,i)),1),'bo-');
    xlabel('Each of the 16 Projections');
    ylabel('Average MSE');
    title(strrep([name_of_exp num2str(i) ' Results'],'_',' '));
    legend('Before projection','After projection');
end;







    
    
    
    
    
    
    
    
    
    
    
 
% ===== EOF ====== [generate_quant_results_automagically.m] ======  
