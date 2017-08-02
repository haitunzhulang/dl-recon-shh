import argparse

# change round to what round of generation we are, if round ==1 then 
# we increase start_index by num_add
# v37
# nohup python3 generate_test_results.py --mindex 33 --theta 60 --start_index 7501 --num_add 200 --data_dirname ../xct-parallelbeam-matlab/dataset_v37_60D_Nonneg_NonIC_noNoise/ --round 0 2>&1&
# nohup python3 generate_test_results.py --mindex 33 --theta 60 --start_index 7701 --num_add 800 --data_dirname ../xct-parallelbeam-matlab/dataset_v37_60D_Nonneg_NonIC_noNoise/ --round 0 2>&1&
# 25265 pid - generating up to 8500

# v38
# nohup python3 generate_test_results.py --mindex 31 --theta 100 --start_index 7501 --num_add 200 --data_dirname ../xct-parallelbeam-matlab/dataset_v38_100D_Nonneg_NonIC_noNoise/ --round 0 2>&1&
# nohup python3 generate_test_results.py --mindex 31 --theta 100 --start_index 7701 --num_add 300 --data_dirname ../xct-parallelbeam-matlab/dataset_v38_100D_Nonneg_NonIC_noNoise/ --round 0 2>&1&

# v39
# nohup python3 generate_test_results.py --mindex 35 --theta 140 --start_index 7701 --num_add 300 --data_dirname ../xct-parallelbeam-matlab/dataset_v39_140D_Nonneg_NonIC_noNoise/ --round 0 2>&1&
# 23235

# v40
# nohup python3 generate_test_results.py --mindex 36 --theta 60 --start_index 7701 --num_add 300 --data_dirname ../xct-parallelbeam-matlab/dataset_v40_60D_Nonneg_NonIC_Noisey/ --round 0 2>&1&
# 23344

# v41
# nohup python3 generate_test_results.py --mindex 37 --theta 100 --start_index 7701 --num_add 300 --data_dirname ../xct-parallelbeam-matlab/dataset_v41_100D_Nonneg_NonIC_Noisey/ --round 0 2>&1&
# 23481

# v42
# nohup python3 generate_test_results.py --mindex 38 --theta 140 --start_index 7701 --num_add 300 --data_dirname ../xct-parallelbeam-matlab/dataset_v42_140D_Nonneg_NonIC_Noisey/ --round 0 2>&1&


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mindex", type=int, default=33)
    parser.add_argument("--data_dirname", type=str, default='../xct-parallelbeam-matlab/dataset_v37_60D_Nonneg_NonIC_noNoise/')
    parser.add_argument("--theta", type=int, default=60)
    parser.add_argument("--start_index", type=int, default=7501)
    parser.add_argument("--num_add", type=int, default=1500)
    parser.add_argument("--round", type=int, default=0)
    args = parser.parse_args()
    return args

args = get_args()
print(args)


mindex=args.mindex
data_dirname=args.data_dirname
theta=args.theta
num_add=args.num_add
start_index=args.start_index + args.round*num_add


def generate_test_results(mindex,data_dirname,theta,start_index=7501,num_add=1500,mf='projection_results7/',mname='2000_simple'):
    import numpy as np
    import glob
    from PIL import Image
    import helper_functions as hf
    import time

    import matplotlib
    matplotlib.use('TkAgg')
    # from matplotlib import pylab as pl
    # from matplotlib.backends.backend_agg import FigureCanvasAgg
    import importlib
    import scipy.io as sio


    # mf = 'projection_results7/'
    mindex_true = mindex
    # mname = '2000_simple'
    model = hf.load_trained_CNN2_with_mindex(mf=mf,mindex_true=mindex_true,mname=mname)


    import CNN_generator as cg
    # from multi_gpu import make_parallel
    #model = cg.residual_projectionNet(depth=3,k1=16,k2=1,k3=8,nb_filters=325,input_shape=(256,256,1),longer=0,dropout=0)
    #model = make_parallel(model,8)
    #model.load_weights(mf + models[mindex] + '/best_simple.h5')


    # import CNN_generator as cg
    #model2 = cg.linked_projection_network(num_stacks=0,nb_filters=325,k1=16,k2=1,k3=8,input_shape=(256,256,1))
    model2=cg.residual_projectionNet(depth=20,k1=16,k2=1,k3=8,nb_filters=64,input_shape=(256,256,1),dropout=0)
    model2.set_weights(model.get_weights())


    #data_dirname='../xct-parallelbeam-matlab/dataset_v38_100D_Nonneg_NonIC_noNoise/'
    # data_dirname='../xct-parallelbeam-matlab/dataset_v37_60D_Nonneg_NonIC_noNoise/'
    # data_dirname='../xct-parallelbeam-matlab/dataset_v31_140_noIC_SL_Attempt_no_Noise/'
    #data_dirname='../xct-parallelbeam-matlab/dataset_v25f_60_Noise_05_noIC/'
    #data_dirname='../xct-parallelbeam-matlab/dataset_v19_100_noRI_scale_nonneg_noInvCrime_PoissonNoise/'
    #dataset_v23f_60_noRI_scale_nonneg_noInvCrime/'

    mindex=mindex_true # 1000 -> just getting results for each of our datasets to start running TV penalty stuff for them
    # theta=100
    niters=10000
    cutoff=.001
    linked=False
    display=False
    num_val = num_add
    PSNR_LS = []
    PSNR_Single_Projection = []
    PSNR_QP_LS = []
    pls=True
    for i in range(num_val):
        index = start_index+i
        fname = 'May2_GetNIPS_'+str(index)
        psnr_recon,psnr_projection,psnr_ls,ls_starting_with_projection = hf.compare_psnr(model2,njumps=10,mindex=mindex,mf=mf,niters=niters,cutoff=cutoff,title='epoch'+mname+' ' +fname+'_'+str(niters)+'_co'+str(cutoff),index=index,version=3,theta=theta,display=display,pls=pls,data_dirname=data_dirname,PSNR=True,linked=linked)
    # PSNR_LS.append(psnr_recon)
    # PSNR_Single_Projection.append(psnr_projection)
    # PSNR_QP_LS.append(psnr_ls)


generate_test_results(mindex,data_dirname,theta,start_index=start_index,num_add=num_add,mf='projection_results7/',mname='2000_simple')





# EoF