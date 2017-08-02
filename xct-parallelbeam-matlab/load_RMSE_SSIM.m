function [MSE_post,MSE_prior,TV_best,Change_Between_posts,Change_Between_priors,Change_From_Projection,SSIM_post,SSIM_prior,SSIM_TV] = load_RMSE_SSIM(result_prefix,result_postfix,support,num_projs,tv_storage,start_index,end_index,TV_included)
% SHOW_PLOTS_LS_TV_SINGLE_MULTIPLE ... 
%  
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 17-Apr-2017 12:34:16 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : show_plots_LS_TV_Single_Multiple.m 


run_fold = zeros(1,end_index-start_index+1);

if TV_included
    TV_best = zeros(length(run_fold),num_projs);
end;



SSIM_post = zeros(end_index,num_projs);
SSIM_prior = zeros(end_index,num_projs);
SSIM_TV = zeros(end_index,num_projs);
MSE_prior = zeros(end_index,num_projs);
MSE_post = zeros(end_index,num_projs);
% Prior_indices = zeros(length(run_fold),num_projs);
% Post_indices = zeros(length(run_fold),num_projs);
DFLoss_prior = zeros(end_index,num_projs);
DFLoss_post = zeros(end_index,num_projs);
% MSE_final = zeros(length(run_fold),1);
Change_Between_posts = zeros(end_index,num_projs);
Change_Between_priors = zeros(end_index,num_projs);
Change_From_Projection = zeros(end_index,num_projs);
for i=start_index:end_index
    %
    if length(dir([tv_storage num2str(i) '.mat'])) > 0 && length(dir([result_prefix num2str(i) result_postfix])) > 0
        load([result_prefix num2str(i) result_postfix]);
        if (mod(i,10) == 0) disp(['On iteration: ' num2str(i)]); end;
        if length(peak_saves_iterations)==2
            len =1;
        else
            len = length(peak_saves_iterations)/2-1;
        end;
        f_true = double(reshape(f_true,[256,256]));

        if TV_included
            [tmp_tv,TV_best(i,1)] = acquire_best_TV(g,f_true,[0 1],support,i,tv_storage);
            SSIM_TV(i,1) = ssim(tmp_tv(support(1):support(2),support(3):support(4)),f_true(support(1):support(2),support(3):support(4)));
    %                 clf;
    %                 subplot(1,2,1);
    %                 imagesc(tmp_tv);
    %                 subplot(1,2,2);
    %                 imagesc(squeeze(peak_saves(1+(1-1)*2,:,:))');
    %                 drawnow();
    %                 pause();
            for j=2:num_projs
                TV_best(i,j)=TV_best(i,1);
                SSIM_TV(i,j)=SSIM_TV(i,1);


            end;
        end;


        % Save MSE pre and post projection for each projection
        for j=1:len

            DFLoss_prior(i,j) = data_fidelity_loss(peak_saves_iterations(1+(j-1)*2)+1);
            DFLoss_post(i,j) = data_fidelity_loss(peak_saves_iterations(2+(j-1)*2)+1);
    %             clf;
    %             subplot(1,2,1);
    %             imagesc(squeeze(peak_saves(1+(j-1)*2,support(1):support(2),support(3):support(4)))');
    %             subplot(1,2,2);
    %             imagesc(f_true(support(1):support(2),support(3):support(4)));
    %             drawnow();
    %             pause();

    %         Prior_indices = peak_saves_iterations(1+(j-1)*2)+1;
    %         Post_indices = peak_saves_iterations(2+(j-1)*2)+1;

            if iscell(peak_saves)
                first = double(reshape(peak_saves{1+(j-1)*2},[256 256]));
                second = double(reshape(peak_saves{2+(j-1)*2},[256 256]));
            else
                first = double(squeeze(peak_saves(1+(j-1)*2)));
                second = double(squeeze(peak_saves(2+(j-1)*2)));
            end;
            first(first<0) =0;
            second(second<0)=0;
            MSE_prior(i,j) = immse(first(support(1):support(2),support(3):support(4))',f_true(support(1):support(2),support(3):support(4)));
            MSE_post(i,j) = immse(second(support(1):support(2),support(3):support(4))',f_true(support(1):support(2),support(3):support(4)));
            Change_From_Projection(i,j) = immse(first(support(1):support(2),support(3):support(4)),second(support(1):support(2),support(3):support(4)));

            SSIM_prior(i,j) = ssim(first(support(1):support(2),support(3):support(4))',f_true(support(1):support(2),support(3):support(4)));
            SSIM_post(i,j) = ssim(second(support(1):support(2),support(3):support(4))',f_true(support(1):support(2),support(3):support(4)));

            if j>1
                Change_Between_posts(i,j)=immse(second(support(1):support(2),support(3):support(4)),old_second(support(1):support(2),support(3):support(4)));
                Change_Between_priors(i,j) = immse(first(support(1):support(2),support(3):support(4)),old_first(support(1):support(2),support(3):support(4)));
            end;
            old_second = second;
            old_first = first;

    %             MSE_prior(i,j) = psnr_save(peak_saves_iterations(1+(j-1)*2)+1);
    %             Prior_indices = peak_saves_iterations(1+(j-1)*2)+1;
    %             Post_indices = peak_saves_iterations(2+(j-1)*2)+1;
    %             MSE_post(i,j) = psnr_save(peak_saves_iterations(2+(j-1)*2)+1);
        end;
    %         MSE_final(i) = psnr_save(end);
    %     if len >=num_projs
    %         ppls = squeeze(peak_saves(1+(j-1)*2,:,:));
    %     else
    %         ppls = double(reshape(ppls,[256 256]))';
    %     end;

    %         ppls = reshape(ppls,[256 256]);
    %          clf;
    %         subplot(1,2,1);
    %         imagesc(ppls(support(1):support(2),support(3):support(4))');
    %         subplot(1,2,2);
    %         imagesc(f_true(support(1):support(2),support(3):support(4)));
    %         drawnow();
    %         pause();

    %     MSE_final(i) = immse(ppls(support(1):support(2),support(3):support(4))',f_true(support(1):support(2),support(3):support(4)));
    end;
    
end;
%     if start_index >1 || end_index < length(run_fold)
%         return;
%     end;

% Show table
% Num_Samples = sum(MSE_post~=0,1);
% Prior_avg = sum(MSE_prior,1)./Num_Samples;
% Post_avg = sum(MSE_post,1)./Num_Samples;
% Prior_loss_avg = sum(DFLoss_prior,1)./Num_Samples;
% Post_loss_avg = sum(DFLoss_post,1)./Num_Samples;
% Improvement = mean(MSE_prior,1)./Num_Samples-mean(MSE_post,1)./Num_Samples;
% PSNR_Improvement = convert_mse_to_psnr(mean(MSE_post,1))-convert_mse_to_psnr(mean(MSE_prior,1));
% if TV_included
%     TV_best_avg = mean(TV_best,1);
% end;








% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [show_plots_LS_TV_Single_Multiple.m] ======  
