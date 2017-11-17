% GENERATE_QUANT_RESULTS for the experiments ... 
%  
%  1.  Showcase the loss plots from the multiple runs.
%  2.   
%

%% Author    : Shenghua He <shenghuahe@wustl.edu> 
%% Date      : 12-Sep-2017 13:54:58 
%% Revision  : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : sh_generate_quant_results.m


%% 1.1 Showing some visuals of images and their corresponding MSE plots 
name_of_exp = 'Experiment_4_64_'; 
start_index = 751; % validation image
end_index = 900;

results_before = zeros([1 31]);
results_after = zeros([1 31]);

figure(6);
clf;
set(gcf,'Position',[1297 222 436 737]);

figure(7);
clf;
[ha1,pos] = tight_subplot(5,13,[.03 .005],[.01 .01],[.001 .001]);
set(gcf,'Position',[2 223 1293 736]);


% for index=start_index:end_index
    index = 800;
    stage=0;
    pickle_file = ['../dl-limitedview-prior/datasets/v' ...
        name_of_exp num2str(0) '_AD' ...
        '/' num2str(index) '.pkl' ];
    M = loadpickle(pickle_file);
    mean_squared_error = mean(mean((M.X_train-M.y_train).^2,2),3);
    i = 2;
    results_before(1+(index-start_index)*10+(i-1),:) = mean_squared_error(1+62*(i-1):2:1+62*i-2);
    results_after(1+(index-start_index)*10+(i-1),:) = mean_squared_error(2+62*(i-1):2:2+62*i-2);

    figure(6);
    clf;
    [ha,pos] = tight_subplot(2,1,[.1 .05],[.01 .05],[.3 .05]);
    axes(ha(1));
    hold on;
    plot(results_before(1+(index-start_index)*10+(i-1),:),'ro-');
    hold on;
    plot(results_after(1+(index-start_index)*10+(i-1),:),'bo-');
    xlabel('Each of the 16 projections');
    ylabel('MSE');
    legend('Before projection','After projection');
    title(['Validation example:' num2str(index*10 + i)]); 

    axes(ha(2));
    imagesc(squeeze(M.y_train((i-1)*62+1,:,:)));
    title('Target Image');
    axis off;
    
    figure(7);
    for j=1:62
        axes(ha1(j));
        imagesc(squeeze(M.X_train((i-1)*62+j,:,:)));
        axis off;
    end;
%     figure(1)
%     plot(results_before, '-ro')
%     hold on
%     plot(results_after, '-bo')
    
%     if exist(pickle_file,'file')>0
%         M = loadpickle(pickle_file);
%         if size(M.X_train,1) < 320
%             error(['file:datasets/v' name_of_exp num2str(stage) '_AD/' num2str(index) '.pkl did not have 320 items.  it had: ' num2str(size(M,1))]);
%         end;
%         
%         mean_squared_error = mean(mean((M.X_train-M.y_train).^2,2),3);
%         size(mean_squared_error)
%         for i=1:10
%             results_before(1+(start_index-index)*10+(i-1),:) = mean_squared_error(1+62*(i-1):2:1+62*i-2);
%             results_after(1+(start_index-index)*10+(i-1),:) = mean_squared_error(2+62*(i-1):2:2+62*i-2);
%             size(results_before)
%                         
%             figure(7);
%             for j=1:32
%                 axes(ha1(j));
%                 imagesc(squeeze(M.X_train((i-1)*32+j,:,:)));
%                 axis off;
%             end;
%             
%             figure(6);
%             clf;
%             [ha,pos] = tight_subplot(2,1,[.1 .05],[.01 .05],[.3 .05]);
%             axes(ha(1));
%             hold on;
%             plot(results_before(1+(start_index-index)*10+(i-1),:),'ro-');
%             hold on;
%             plot(results_after(1+(start_index-index)*10+(i-1),:),'bo-');
%             xlabel('Each of the 16 projections');
%             ylabel('MSE');
%             legend('Before projection','After projection');
%             title(['Validation example:' num2str(index*10 + i)]); 
% %             axis on;
%             
%             axes(ha(2));
%             imagesc(squeeze(M.y_train((i-1)*32+1,:,:)));
%             title('Target Image');
%             axis off;
%             
%             disp(['Just plotted for ' num2str(index*10 + i)]);
%             drawnow();
%             pause();
%         end;
%     end;
    
%     
% end; 