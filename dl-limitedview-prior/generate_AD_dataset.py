# scp -r bmkelly@turing.seas.wustl.edu:/home/bmkelly/dl-limitedview-prior/generate_AD_dataset.py .

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
from keras.callbacks import Callback
import helper_functions as hf
import CNN_generator as cg
import numpy as np
import argparse
import random
from keras.optimizers import SGD,Adam
import contextlib

import math


# python3 generate_AD_dataset.py --dataset 19 --save_dir v19_AD

# old_env = os.environ["CUDA_VISIBLE_DEVICES"]





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f_dim1", type=int, default=16)
    parser.add_argument("--f_dim2", type=int, default=1)
    parser.add_argument("--f_dim3", type=int, default=8)
    parser.add_argument("--nb_filters", type=int, default=325)
    parser.add_argument("--dataset",type=int,default=13)
    parser.add_argument("--num_stacks",type=int,default=1)
    parser.add_argument("--tmp_save",type=int,default=0)
    parser.add_argument("--num_add",type=int,default=50)
    parser.add_argument("--model_folder",type=str,default='trained_v19')
    parser.add_argument("--save_dir",type=str,default='tmp/')
    args = parser.parse_args()
    return args

args = get_args()


name = args.model_folder
#name = 'deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_50_False_0.0_True_False_13_False_5_False_500_0.5_1.0_10000_num_stacks1_march13_linked'
# mkdir trained_v19/
# cp projection_results5/deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_19_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_old/best_simple.yaml trained_v19/best_simple.yaml
# cp projection_results5/deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_19_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_old/best_simple.h5 trained_v19/best_simple.h5
# scp -r bmkelly@turing.seas.wustl.edu:/home/bmkelly/dl-limitedview-prior/trained_v19 .
# Load model
model_just_weights = hf.load_trained_CNN(load_weights=True,name=name+'/best_simple',folder='')
model_larger = cg.linked_projection_network(input_shape=(256,256,1),k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters,num_stacks=args.num_stacks)
model_larger.set_weights(model_just_weights.get_weights())


if args.dataset == 13:
    max_index=8000
    theta=60
    data_dirname='../xct-parallelbeam-matlab/dataset_v13_60_noRI_scale_nonneg/'
    H_DIRNAME = '../xct-parallelbeam-matlab/system-matrix/'
elif args.dataset==18:
    max_index=9000
    theta=60
    data_dirname='../xct-parallelbeam-matlab/dataset_v17_100_noRI_scale_nonneg_noInvCrime_PoissonNoise/'
    H_DIRNAME = '../xct-parallelbeam-matlab/system-matrix/'
elif args.dataset==19:
    max_index=7000
    theta=100
    data_dirname='xct-parallelbeam-matlab/dataset_v19_100_noRI_scale_nonneg_noInvCrime_PoissonNoise/'
    H_DIRNAME = 'xct-parallelbeam-matlab/system-matrix/'
            
# scp -r bmkelly@deeplearning.seas.wustl.edu:/home/bmkelly/xct-parallelbeam-matlab/system-matrix xct-parallelbeam-matlab
# scp -r bmkelly@deeplearning.seas.wustl.edu:/home/bmkelly/xct-parallelbeam-matlab/dataset_v19_100_noRI_scale_nonneg_noInvCrime_PoissonNoise  xct-parallelbeam-matlab


# Generate datapoints, save to pkl file
def generate_AD(save_file=args.save_dir + str(args.tmp_save) +'.pkl' ,model_larger=model_larger,num_add=args.num_add,max_index=max_index,theta=theta,data_dirname=data_dirname,H_DIRNAME=H_DIRNAME):
    Y_train = []
    X_train = []

    for k in range(num_add):
        index = random.randint(0,max_index)
        skip=False
        try:
            H,g,f_true,f_recon = hf.load_H_g_target(version=2,index=index,theta=theta,dirname=data_dirname,H_DIRNAME=H_DIRNAME)
        except Exception:
            print('couldnt find index: ' + str(index))
            skip=True
        if not skip:
            cutoff=np.random.random()*.02 #max cutoff is .01
            niters= math.floor(np.random.random()*5000)
            ls_starting_with_projection,psnr_save,projection_times_save,magnitudes_save = hf.Projected_PLS(model_larger,cutoff=cutoff,niters=niters,version=2,theta=theta,index=index,display=False,data_dirname=data_dirname,linked=True,H_DIRNAME=H_DIRNAME)
            
            Y_inter = []
            X_inter = []
            X_inter.append(ls_starting_with_projection.reshape(256,256,1))
            Y_inter.append(f_true.reshape(256,256,1))
            X_inter = np.asarray(X_inter)
            Y_inter = np.asarray(Y_inter)
            #X_inter = fix_Ys(X_inter,model.input_shape)
            #Y_inter = fix_Ys(Y_inter,model.output_shape[0])
    
            X_train.append(X_inter[0])
            Y_train.append(Y_inter[0]) # hopefully this is the same as Y_train[index]...


    # Save Data
    import pickle
    data = {}
    data['X_train'],data['y_train'] = X_train,Y_train
    with open(save_file, 'wb') as f:
        pickle.dump(data,f)

generate_AD()







# EoF