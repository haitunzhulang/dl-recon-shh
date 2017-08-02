function NIPSWRAPUP_Compare_Ending_R_Q()
% NIPSWRAPUP_COMPARE_ENDING_R_Q ... 
%  
%  

%% Author    : Brendan Kelly <bmkelly@wustl.edu> 
%% Date     : 26-May-2017 09:41:47 
%% Revision : 1.00 
%% Developed : 9.1.0.441655 (R2016b) 
%% Filename  : NIPSWRAPUP_Compare_Ending_R_Q.m 

mf_mat = '/home/bmkelly/dl-limitedview-prior/NIPS_2017_DL-Guided_Backup/output_transfer_for_matlab/';


mf = '/home/bmkelly/xct-parallelbeam-matlab/NIPS_2017_Work_Backup/dataset_v38_100D_Nonneg_NonIC_noNoise/';
result_prefix = [mf_mat '31_'];
result_postfix = ['_10000_0.001.mat'];
tv_storage ='/home/bmkelly/xct-parallelbeam-matlab/NIPS_2017_Work_Backup/tmp_grid_v38/';
super_title = '100D Model Error, Noiseless';

% mf = '/home/bmkelly/xct-parallelbeam-matlab/NIPS_2017_Work_Backup/dataset_v39_140D_Nonneg_NonIC_noNoise/';
% result_prefix = [mf_mat '35_'];
% result_postfix = ['_10000_0.001.mat'];
% tv_storage ='/home/bmkelly/xct-parallelbeam-matlab/NIPS_2017_Work_Backup/tmp_grid_v39/';
% super_title = '140D Model Error, Noiseless';


num_projections=5;


b=40;
support = [1+b,256-b,1+b,256-b];

start_index=7515;


for i=start_index:8000
    load([result_prefix num2str(i) result_postfix]);
    single_pass = squeeze(peak_saves{2})';
    our_approach = squeeze(peak_saves{10})';
    end_Q = squeeze(peak_saves{10})';
    end_R = squeeze(peak_saves{11})';
    single_pass(single_pass<0) = 0; our_approach(our_approach<0)=0;
    end_Q(end_Q<0) = 0; end_R(end_R<0) = 0;
    % Original
    [recon_img, true_img, g] = load_data_given_index_mf(mf,i);
    true_img = reshape(true_img,[256 256]);
    recon_img = reshape(recon_img,[256 256]);
%     recon_img(recon_img<0) = 0;
    
    % TV
    fname = [ num2str(i) '.mat'];
    load([tv_storage fname]);
    [optimal_mse,optimal_index] = min(mses);
    optimal_image = squeeze(img_recons(optimal_index,:,:));
    dif1 = true_img-recon_img;dif3= true_img-single_pass;
    dif2= true_img-optimal_image;dif4=true_img-our_approach;
    dif_scale = [min([min(min(dif1)) min(min(dif2)) min(min(dif3)) min(min(dif4))]) ...
                max([max(max(dif1)) max(max(dif2)) max(max(dif3)) max(max(dif4))])];
    
            
    
    min_color = min([min(true_img(:)) min(end_Q(:)) min(end_R(:))]);
    max_color = max([max(true_img(:)) max(end_Q(:)) max(end_R(:))]);
    clf;
    set(gcf,'color','w');
%     set(gca,'visible','off');
    [ha, pos] = tight_subplot(1, 3, [.025 .01], [.025 .05] , [.01 .01]);
    axes(ha(1));
    imagesc(true_img(support(1):support(2),support(3):support(4)));
    title('Original Image');
    caxis([min_color max_color]);
    box off; set(gca,'xcolor',get(gcf,'color'));set(gca,'xtick',[]);set(gca,'ycolor',get(gcf,'color'));set(gca,'ytick',[]);
    
    axes(ha(2));
    imagesc(end_Q(support(1):support(2),support(3):support(4)));
    title('Ending with Q - CNN Projection Operator');
    caxis([min_color max_color]);
    box off; set(gca,'xcolor',get(gcf,'color'));set(gca,'xtick',[]);set(gca,'ycolor',get(gcf,'color'));set(gca,'ytick',[]);
    
    axes(ha(3));
    imagesc(end_R(support(1):support(2),support(3):support(4)));
    title('Ending with R -- Least Squares Optimization');
    caxis([min_color max_color]);
    box off; set(gca,'xcolor',get(gcf,'color'));set(gca,'xtick',[]);set(gca,'ycolor',get(gcf,'color'));set(gca,'ytick',[]);
    
    disp([num2str(i)]);
    
    drawnow();
    pause();
    
end; 



% ===== EOF ====== [NIPSWRAPUP_Compare_Ending_R_Q.m] ======  
