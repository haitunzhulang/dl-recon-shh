function acquire_best_TV_grid_search(g,f_true,H,support,index,tv_storage,test)
% ACQUIRE_BEST_TV_GRID_SEARCH ... 
%  
%   ... 

%% AUTHOR    : Frank Gonzalez-Morphy 
%% $DATE     : 09-May-2017 16:24:03 $ 
%% $Revision : 1.00 $ 
%% DEVELOPED : 9.1.0.441655 (R2016b) 
%% FILENAME  : acquire_best_TV_grid_search.m 

%%
if ~exist('test')
    test=0;
end;

% keyboard();
% Try to load if exists
fname = [tv_storage num2str(index) '.mat'];
previous_calculation = dir(fname);
VERBOSE=0;
DEBUG=1;

if length(previous_calculation)==1
    load([fname]);
    [Y,I] = min(mses);
    best_mse = Y;
    img = squeeze(img_recons(I,:,:));
    if VERBOSE
        [Y,I] = sort(reg_params,'descend');
        for i=1:length(reg_params)
            subplot(4,4,i);
            imagesc(squeeze(img_recons((i),:,:)));
            title(['W: ' num2str(reg_params(I(i))) '. MSE: ' num2str(mses(I(i)))]);
        end;
        subplot(4,4,16);
        semilogx(reg_params(I),mses(I),'^-');
        title('Mean squared errors vs Weight Parameter');
        xlabel('Weight Parameter');
        ylabel('MSE')
        axis([-inf inf min(mses(I(2:end)))*.9 max(mses(I(2:end)))*1.1]);
        drawnow();
        pause();
    end;
    return;
end;

data.H = H;
data.g = double(g(:));


cost_cutoff=.001;
STEP_SIZE=.75;
MAX_ITER=300;
NX=256;
NY=256;

drawing = 0;
reg_params = logspace(-2,-9,140);
if test 
    reg_params=[.01];
end;
mses = zeros(1,length(reg_params));
img_recons = zeros([length(reg_params),256,256]);
tic;
for i=1:length(reg_params)
    img_recons(i,:,:) = fistatv2d(@cost_func_xray_H, zeros(NX,NY), data, ...
            STEP_SIZE, reg_params(i), ...
        'output_filename_prefix', '', 'verbose', VERBOSE, ...
        'min_rel_cost_diff', cost_cutoff,...
        'max_iter', MAX_ITER, 'proj_op','nonneg');
    mses(i) = immse(squeeze(img_recons(i,support(1):support(2),support(3):support(4))),f_true(support(1):support(2),support(3):support(4)));
    if i==1
        disp(['Finished the first reg param, out of ' num2str(length(reg_params))]);
        toc;
    end;
end;



[Y,I] = min(mses);
best_mse = Y;
img = squeeze(img_recons(I,:,:));

save(fname,'img', 'best_mse','img_recons','mses','reg_params');






% Created with NEWFCN.m by Frank Gonz�lez-Morphy  
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [acquire_best_TV_grid_search.m] ======  
