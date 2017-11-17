# scp bmkelly@turing.seas.wustl.edu:/home/bmkelly/dl-limitedview-prior/helper_functions.py .
from __future__ import print_function
import os
from io import BytesIO
import numpy as np
from functools import partial
import time
import math
import pickle
import sys
import scipy.io

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D
from keras.utils import np_utils
from keras.models import model_from_yaml
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping,LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
KTF.set_session(sess)


def load_trained_CNN(version=1,load_weights=True,name=None,deploy=False,weight_file=None,folder='models/'):
    if name is not None:
        model_name = folder + name
    else:
        if version==1:
            model_name = 'models/jan18_regression_v3_densenet'
    
        if version==2:
            model_name = 'models/jan18_regression_v4_5_1_mse_densenet'
    
        if version==3: 
            model_name = 'models/model_dropout_regression_jan13'
    
    if deploy:
        yaml_file = open(model_name+'_deploy.yaml', 'r')
    else:
        yaml_file = open(model_name+'.yaml', 'r')
    
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    loaded_model = model_from_yaml(loaded_model_yaml)
    
    
    if load_weights:
        if weight_file is not None: # Specific weight file
            loaded_model.load_weights(weight_file)
        else:
            loaded_model.load_weights(model_name+".h5")
    
    model = loaded_model
    print("Loaded model from disk")
    return model
    
def load_trained_CNN2_with_mindex(mf='projection_results7/',mindex_true=29,mname='best_simple'):
    
    # mar 2
    models = []
    models.append('deartifact_False_0.300_16_1_8_325_False_0.01_mse_2_100_False_0.0_True_False_13_False_0_looping_march1')
    models.append('deartifact_False_0.300_16_1_8_325_False_0.01_mse_2_100_False_0.0_True_False_13_False_looping_march1')
    # mar 7
    models.append('deartifact_False_0.300_16_1_8_64_False_0.01_mse_0_100_False_0.0_True_False_13_False_0_looping_march1')
    models.append('deartifact_False_0.300_16_1_8_64_False_0.01_mse_0_100_False_0.0_True_False_13_False_10_looping_march1')
    # mar 8
    models.append('deartifact_False_0.300_16_1_8_325_False_0.001_mse_0_100_False_0.0_True_False_13_False_0_True_ 500_0.5__march7')
    models.append('deartifact_False_0.300_16_1_8_64_False_0.01_mse_0_100_False_0.0_True_False_13_False_10_looping_march1')
    # mar 9
    models.append('deartifact_False_0.300_16_1_8_325_False_0.001_mse_0_100_False_0.0_True_False_13_False_5_True_ 500_0.5_0.1__march8')
    # mar 13
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_50_False_0.0_True_False_13_False_5_False_500_0.5_1.0_150_march8')
    #mar 14
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_50_False_0.0_True_False_13_False_5_False_500_0.5_1.0_1000_march13_linked')
    #mar 15
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_20_False_0.0_True_False_13_False_5_False_500_0.5_1.0_1000_num_stacks1_equalADTrue_num_normal_training_examples1000_march13_linked')
    # mar 17
    models.append('')
    # mar 19 -- best AD CNN for dataset 13
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_20_False_0.0_True_False_13_False_5_False_500_0.5_1.0_1000_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_simple')
    mindex=11
    # Mar 21
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_19_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_old')
    #mindex = 12

    # Mar 24
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_13_False_0_False_500_0.5_1.0_num_stacks1_batchsize_32_march13_linked')
    mindex=13

    # mar 26
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_23_False_0_False_500_0.5_1.0_num_stacks1_batchsize_32_march13_linked')
    mindex=14
    # old
    models.append('deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_19_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_old')
    mindex=15

    # apr 5
    models.append('deartifact__16_1_8_325_0.001_mse_0_100_5_0.0_23_False_500_batchsize_28_cycling_True_minlr_5e-05_maxlr_0.0005_lrss_20_march13_linked')
    mindex=16

    # apr 10 - v19 AD
    models.append('deartifact__16_1_8_325_0.0005_mse_0_100_5_0.0_19_batchsize_64_cycling_False_minlr_0.0005_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_march13_linked')
    mindex=17

    # apr 10 v25 Pre AD
    models.append('deartifact__16_1_8_325_0.001_mse_0_100_5_0.0_25_batchsize_64_cycling_False_minlr_0.001_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_march13_linked')
    mindex=18

    # apr 11 v25 - AD
    models.append('deartifact__16_1_8_325_0.001_mse_0_100_5_0.0_25_batchsize_64_cycling_False_minlr_0.001_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_apr11')
    mindex=19

    # apr 13 v19 - AD
    models.append('deartifact__16_1_8_325_0.0001_mse_0_100_5_0.0_19_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_apr11')
    # mindex=20

    # Apr 19 - v31 - No AD
    models.append('deartifact__16_1_8_325_lr0.0001_mse_numEpochs_4000_DO0.0_v31_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_3_apr14')
    mindex = 21

    models.append('deartifact__16_1_8_325_lr0.0001_mse_numEpochs_4000_DO0.5_v31_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_3_apr14')
    mindex=22

    # Apr 21 - v31 - AD
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v31_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr21')
    mindex=23
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v31_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=24
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v31_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_4_AD_True_apr22')
    mindex=25

    # Apr 24 - v312 - AD+
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v312_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=26

    mindex =24

    # Apr 27 - v35 Reg
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v35_batchsize_60_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_False_apr22')
    mindex=27

    # May 2 - v35 AD
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v35_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=28

    # May 2 - v352 AD
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v352_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=29
    
    # May 3 - v352 AD - AGAIN -- Trained for 2k iterations
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v352_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=30
    
    # May 5 - v38 AD
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v38_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=31
    
    # May 5 - v36 Reg
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v36_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_False_apr22')
    mindex=32
    
    # May 8 - v37 AD 
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v37_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=33
    
    # May 8 - v37 REG 
    models.append('deartifact__16_1_8_64_lr0.000102_mse_numEpochs_4000_DO0.0_v37_batchsize_60_cycling_False_minlr_0.000102_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_False_apr22')
    mindex=34
    
    # May 9 - v39 AD 
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v39_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=35
    
    # v40
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v40_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=36
    
    # v41
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v41_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=37
    
    # v42
    models.append('deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v42_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_True_apr22')
    mindex=38
    
#     mf = 'projection_results7/'#'transfer_models/models/'#
    nname = models[mindex_true]
#     mname = 'best_simple'
    model = load_trained_CNN(name=mf + nname + '/' + mname,folder='')#+ '/' + mname ,folder='')
    return model


def load_trained_CNN_Regression_jan13():
    img_dim = (256,256)
    img_width,img_height = img_dim[0],img_dim[1]
    
    input_shape = (img_dim[0], img_dim[1], 1)
    img_rows, img_cols = img_dim[0],img_dim[1]# number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    pool_size = (2, 2)
    # convolution kernel size
    kernel_size = (3, 3)

    dropout = .5
    conv_count = 1

    def add_module(model,cc,dropout=.5):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],name ='conv'+str(cc)))
        model.add(Activation('relu'))
        return cc+1

    model = Sequential()
    model.add(ZeroPadding2D((0, 0), batch_input_shape=(1, img_width, img_height,1)))
    first_layer = model.layers[-1]
    # # this is a placeholder tensor that will contain our generated images
    input_img = first_layer.input
    # Layer 1
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape,name = 'conv'+str(conv_count)))
    model.add(Activation('relu'))

    conv_count+=1

    # Layer 2
    for i in range(1):
        conv_count=add_module(model,conv_count,dropout)

    model.add(MaxPooling2D(pool_size=pool_size))

    # Layer 3-4
    for i in range(2):
        conv_count=add_module(model,conv_count,dropout)

    model.add(MaxPooling2D(pool_size=pool_size))

    # 5-6
    for i in range(2):
        conv_count=add_module(model,conv_count,dropout)

    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Convolution2D(32, kernel_size[0], kernel_size[1],name ='conv'+str(conv_count)))
    model.add(Activation('relu'))
    conv_count+=1
    model.add(GlobalAveragePooling2D())
    model.add(Dense(128))

    model.add(Dense(1, init='normal'))

    model.load_weights("models/model_dropout_regression_jan13.h5")
    return model,input_img
    
    
def compute_derivative(model,img0,input_img,filter_index,layer_name='softmax',output_2=True):
    layer_dict = dict([(layer.name, layer) for layer in model.layers])
    layer_output = layer_dict[layer_name].output
    
    # minimize the mean for a specific node in layer_output
    if output_2:
        loss = K.mean(layer_output[:, filter_index])
    else:
        loss = K.mean(layer_output[:,:,:,filter_index])
    # compute gradient wrt the image
    grads = K.gradients(loss, input_img)[0]
    # normalization trick: we normalize the gradient -- dont do this.
#     grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)
    
    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads])
    step = 1
    return iterate([img0])
    

def predict_outputs(model,X,target_dim = (256,256,1)):
    outputs = []
    sz = np.shape(X)
    
    for i in range(sz[0]):
        res = model.predict(np.reshape(X[i,:],[1,target_dim[0],target_dim[1],target_dim[2]]))
        outputs.append(res[0][0])
        
    return outputs
   
# preds -> predictions from ml algorithm [x,]
# true_labels -> true regression targets [x,]
# ml_name -> Name of algorithm to be used in titles/legends  
def display_distribution_of_outputs_regression(preds,true_labels,ml_name,display=True,output_folder=None,name_extra=''):
    
    true_sorted = np.sort(true_labels)
    indices = np.argsort(true_labels)

    outs_sorted = np.sort(preds)
    
    # Draw predictions
    if display is False:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(10,10))
    else:
        from matplotlib import pylab as pl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        fig = pl.figure(figsize=(8,8))
        pl.ion()
    
    x = list(range(1,len(outs_sorted)+1))
    
    #mse = ((np.array(preds)[indices] - np.array(true_labels)[indices]) ** 2).mean(axis=0)
    
    fig.clf()
    ax = fig.add_subplot(2,2,1)
    ax.plot(x,np.array(preds)[indices])
    ax.plot(x,np.array(true_labels)[indices])
    ax.set_ylabel('Regression Target')
    ax.legend(['Probabilistic Output from CNN',' Correct regression output'],loc='lower right',prop={'size':12})
    ax.set_title('Probability Scores for Artifact Classification')
    

    # Draw Distribution
#     fig = pl.figure()
#     pl.ion()
#     fig.clf()
    ax = fig.add_subplot(2,2,2)
    ax.hist(np.array(preds))
#     ax.hist(np.array(true_labels))
    ax.set_title("Probability Outputs Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
#     ax.legend(['Probabilistic Output from the Classification CNN','Correct regression output'],loc='upper left',prop={'size':12})
    
    
#     fig = pl.figure()
#     pl.ion()
#     fig.clf()
    ax = fig.add_subplot(2,2,3)
    ax.hist(np.array(preds))
    ax.hist(np.array(true_labels))
    ax.set_title("Probability Outputs Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.legend(['Probabilistic Output from the CNN',' Correct regression output'],loc='upper left',prop={'size':12})
    
    if display:
        pl.show()
    
    
    if output_folder is not None:
        save_folder = 'PLS_Out/'+ output_folder + '/'
        generate_folder(save_folder)    
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(save_folder +'distribution_graph_ '+ name_extra + '.png', dpi=40)
        





def normalize_dataset(X_train,X_test,y_train,y_test,img_dim=(256,256)):
    
    print_len = 1000000
    
    # Normalize
    sz = np.shape(X_train)
    X_train = np.reshape(X_train,(sz[0],img_dim[0],img_dim[1],1))
    
    t=time.time()

    for i in range(sz[0]):
        mmax = np.max(X_train[i,:,:,:])
        mmin = np.min(X_train[i,:,:,:])
        X_train[i,:,:,:] = (X_train[i,:,:,:]-mmin)/(mmax-mmin)

    sz = np.shape(X_test)
    X_test = np.reshape(X_test,(sz[0],img_dim[0],img_dim[1],1))
    t=time.time()
    for i in range(sz[0]):
        mmax = np.max(X_test[i,:,:,:])
        mmin = np.min(X_test[i,:,:,:])
        X_test[i,:,:,:] = (X_test[i,:,:,:]-mmin)/(mmax-mmin)
            
    if len(np.array(y_train).shape) == 4:
        Y_train = y_train
        Y_test = y_test
        sz = Y_train.shape
        t=time.time()
        for i in range(sz[0]):
            mmax = np.max(Y_train[i,:,:,:])
            mmin = np.min(Y_train[i,:,:,:])
            Y_train[i,:,:,:] = (Y_train[i,:,:,:]-mmin)/(mmax-mmin)
            
        sz = Y_test.shape
        t=time.time()
        for i in range(sz[0]):
            mmax = np.max(Y_test[i,:,:,:])
            mmin = np.min(Y_test[i,:,:,:])
            Y_test[i,:,:,:] = (Y_test[i,:,:,:]-mmin)/(mmax-mmin)
        
        return X_train,X_test,Y_train,Y_test
        
    Y_train = np.array(y_train).astype('float32')
    Y_test = np.array(y_test).astype('float32')

    y_min = np.min([np.min(Y_test),np.min(Y_train)])
    y_max = np.max([np.max(Y_test),np.max(Y_train)])
    Y_test = (Y_test-y_min )/(y_max-y_min)
    Y_train= (Y_train-y_min )/(y_max-y_min)
    
    return X_train,X_test,Y_train,Y_test

    

def load_data(version=1,normalize=False,target_shape=(256,256,1),dataset_folder = 'datasets/',normalize_projection=False,normalize_simplistic=False):
    # Version 1 is Regression 100k data
    if version==1:
        fname = 'train_data_regression_100k.pkl'
        final_string = 'regression data 100k-- linear combination of true and recon images.'
        
    # Version 2 is Regression 10k data    
    if version==2:
        fname = 'train_data_regression_10k.pkl'
        final_string = 'regression data 10k-- linear combination of true and recon images.'
       
    
    # Version 3 is Regression - LA Dataset
    if version==3:
        fname = 'train_data_regression_v3.pkl'
        final_string = 'regression data - LA views 120-180.'
        
    # Version 4 is Regression - LA Dataset - actual Regression
    if version==4:
        fname = 'train_data_LA_v2_actual_regression.pkl'
        final_string = 'regression data - LA views 121,122,...,179.'
        
    if version==5:
        fname = 'train_data_regression_10k_noisy.pkl'
        final_string = 'Noisy Regression data - 10k - jan24'
    
    if version==6:
        fname = 'train_deartifacter_80000_crops_64_v6.pkl'
        final_string = 'Projection data - 80k - 64 dimension - feb3'
        target_shape=(64,64,1)
    
    if version==7:
        fname = 'train_deartifacter_14000_crops_196_v7.pkl'
        final_string = 'Projection data - 80k - 196 dimension - feb16'
        target_shape=(196,196,1)
    
    if version==8:
        fname = 'train_deartifacter_40000_crops_196_v8.pkl'
        final_string = 'Projection data - 37k - 196 dimension - feb17'
        target_shape=(196,196,1)
    
    if version==9:
        fname = 'train_deartifacter_9000_crops_196_v9.pkl'
        final_string = 'Projection data - 9000 - 196 dimension - feb17'
        target_shape=(196,196,1)
        
    if version==10:
        fname = 'train_deartifacter_40000_crops_196_v10.pkl'
        final_string = 'Projection data - 9000 - 196 dimension - feb20'
        target_shape=(196,196,1)
    
    if version==11:
        fname = 'train_deartifacter_10000_crops_176_v11.pkl'
        final_string = 'Projection data - 10000 - 176 dimension - feb22'
        target_shape=(176,176,1)
    
    if version==12:
        fname = 'train_deartifacter_8000_crops_176_v12.pkl'
        final_string = 'Projection data - 8000 - 176 dimension - feb24'
        target_shape=(176,176,1)
        
    if version==13:
        fname = 'train_deartifacter_8000_crops_176_v13.pkl'
        final_string = 'Projection data - 8000 - 176 dimension - feb28'
        target_shape=(176,176,1)
        
    if version==16:
        fname = 'train_ssim_pred_via_g_tv_5500_v16.pkl'
        final_string = 'g+tv -> ssim 5500, v16, mar14'
        target_shape=(100,256,1)
        
    if version==19:
        fname = 'train_deartifacter_7000_crops_176_100d_v19.pkl'
        final_string = 'train_deartifacter_7000_crops_176_100d_v19'
        target_shape=(176,176,1)
        
    if version==21:
        fname = 'train_deartifacter_7000_crops_176_60d_v21.pkl'
        final_string = 'train_deartifacter_7000_crops_176_60d_v21'
        target_shape=(176,176,1)
    
    if version==22:
        fname = 'train_deartifacter_7000_crops_176_140d_v22.pkl'
        final_string = 'train_deartifacter_7000_crops_176_140d_v22'
        target_shape=(176,176,1)
    
    if version==23:
        fname = 'train_deartifacter_7000_crops_17660d_v23.pkl'
        final_string = 'train_deartifacter_7000_crops_17660d_v23'
        target_shape=(176,176,1)
        
    if version==25:
        fname = 'train_deartifacter_7500_crops_176_40d_v25.pkl'
        final_string = 'train_deartifacter_7500_crops_176_40d_v25'
        target_shape=(176,176,1)
        
        
        
    
    if version>=1 and version <=25:
         ## Load Data
        import pickle
        with open(dataset_folder + fname, 'rb') as f:
            data = pickle.load(f)
            X_train = data['X_train']
            X_test = data['X_test']
            Y_train = data['y_train']
            Y_test = data['y_test']
            sz = np.shape(X_train)
            X_train = np.reshape(X_train,(sz[0],target_shape[0],target_shape[1],target_shape[2]))
            sz = np.shape(X_test)
            X_test = np.reshape(X_test,(sz[0],target_shape[0],target_shape[1],target_shape[2]))
            if version==16:
                tv_train = np.asarray(data['tv_train']).reshape(np.shape(X_train)[0],1,1,1)
                tv_test = np.asarray(data['tv_test']).reshape(np.shape(X_test)[0],1,1,1)
                Y_train = np.asarray(Y_train)
                Y_test = np.asarray(Y_test)
    
        
    
    if (version >=6 and version <=15) or (version>=18 and version <=25):
        Y_train = np.reshape(Y_train,(np.shape(Y_train)[0],target_shape[0],target_shape[1],target_shape[2]))
        Y_test = np.reshape(Y_test,(np.shape(Y_test)[0],target_shape[0],target_shape[1],target_shape[2]))
      
    
    if normalize + normalize_projection + normalize_simplistic > 1:
        raise ValueError('Too many normalize parameters true, in load_data()')
   
    if normalize_simplistic:
        X_train= X_train*2-1
        Y_train = Y_train*2-1
        X_test = X_test*2-1
        Y_test = Y_test*2-1

    if normalize:
        X_train,X_test,Y_train,Y_test = normalize_dataset(X_train,X_test,Y_train,Y_test)
        # Fix up the LA dataset because it's actually stored so that 120 is 0, and 180 is 1
        if version==3 or version==4:
            Y_train = 1-Y_train
            Y_test = 1-Y_test
        
    if normalize_projection:
        X_train = normalize_data(X_train,mmin=-1,mmax=1,cant_be_0s=True)
        X_test = normalize_data(X_test,mmin=-1,mmax=1,cant_be_0s=True)
        Y_train = normalize_data(Y_train,mmin=-1,mmax=1)
        Y_test = normalize_data(Y_test,mmin=-1,mmax=1)
        
    
    
    print('Done loading ' + final_string)
    if version==16:
        return X_train,X_test,Y_train,Y_test,tv_train,tv_test
    
    return X_train,X_test,Y_train,Y_test

def normalize_single(d_in,mmin=-1,mmax=1):
    dmin = np.min(d_in)
    dmax = np.max(d_in)
    return ((d_in-dmin)/(dmax-dmin))*(mmax-mmin)+mmin

def normalize_data(data_in,mmin=-1,mmax=1,cant_be_0s=False):
    
    data_out = np.zeros(data_in.shape)
    
    for i in range(data_in.shape[0]):
        dmin = np.min(data_in[i,:,:,:])
        dmax = np.max(data_in[i,:,:,:])
        if dmax-dmin==0:
            data_out[i,:,:,:] = mmin
            if cant_be_0s:
                print('Welp it is all 0s... or atleast all the same')
        else:
            data_out[i,:,:,:] = ((data_in[i,:,:,:]-dmin)/(dmax-dmin))*(mmax-mmin)+mmin
    
    return data_out

def run_model_generator(model,dataset='23',nb_epoch=50,batch_size=32,plot_folder ='PLS_Out/',callbacks=[],verbose=1,AD=True,num_gpus=4,non_generator=0):

	val_loss = []

	# checkpoint
	filepath=plot_folder + model.name + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
	callbacks_list = []
	generate_folder(plot_folder + model.name)

	for c in callbacks:
		callbacks_list.append(c)
	
	num_train_samples = 750
	if AD:
		num_val_samples=150
	else:
		num_val_samples=150
	if 0==0 and AD ==True:
		# Load data on the fly
		print('###load data on the fly###')
		tmp_history = model.fit_generator(generator(batch_size=batch_size,input_shape=model.input_shape,output_shape=model.output_shape,dataset=dataset,train=True,AD=AD), nb_epoch=nb_epoch, samples_per_epoch = num_train_samples,
			  verbose=verbose,validation_data=generator(batch_size=batch_size,input_shape=model.input_shape,output_shape=model.output_shape,dataset=dataset,train=False,AD=AD),nb_val_samples=int(num_val_samples/batch_size),callbacks=callbacks_list,pickle_safe=True,nb_worker=2*num_gpus,max_q_size=10*num_gpus)
	else:
		# Load data into memory
		print('###load data into memory###')
		print('###data set:'+dataset)
		X_train,Y_train,X_test,Y_test = generator_load_all(input_shape=model.input_shape,output_shape=model.output_shape,dataset=dataset,AD=AD)
		tmp_history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
					  verbose=verbose, validation_data=(X_test, Y_test),shuffle=True,callbacks=callbacks_list)
	# 
	#     
	#     # Fit the model    
	#     if dataset ==23:
	#         
	# #     if dataset==19:
	# #         tmp_history = model.fit_generator(v19_train(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(7000*10/batch_size), nb_epoch=nb_epoch,
	# #               verbose=verbose,validation_data=v19_test(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(2000/batch_size),callbacks=callbacks_list)
	# 
	#     if dataset==19:
	#         tmp_history = model.fit_generator(v19_train_AD(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(122403*2/batch_size), nb_epoch=nb_epoch,
	#               verbose=verbose,validation_data=v19_test_AD(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(8149/batch_size),callbacks=callbacks_list,pickle_safe=True,nb_worker=20,max_q_size=100)
	# 
	#     
	#     # if dataset==25:
	# #         tmp_history = model.fit_generator(v25_train(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(10408*1/batch_size), nb_epoch=nb_epoch,
	# #               verbose=verbose,validation_data=v25_test(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(2000/batch_size),callbacks=callbacks_list)
	# 
	#     if dataset==25:
	#         tmp_history = model.fit_generator(v25_train_AD(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(143081*2/batch_size), nb_epoch=nb_epoch,
	#               verbose=verbose,validation_data=v25_test_AD(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(8509/batch_size),callbacks=callbacks_list,pickle_safe=True,nb_worker=20,max_q_size=100)
	# 
	# 
	#     if dataset==27:
	#         tmp_history = model.fit_generator(v27_train(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(10408*10/batch_size), nb_epoch=nb_epoch,
	#               verbose=verbose,validation_data=v27_test(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(2000/batch_size),callbacks=callbacks_list)
	# 
	#     if dataset==28:
	#         tmp_history = model.fit_generator(v28_train(batch_size,model.input_shape,model.output_shape), samples_per_epoch = int(7500*10/batch_size), nb_epoch=nb_epoch,
	#               verbose=verbose,validation_data=v28_test(batch_size,model.input_shape,model.output_shape),nb_val_samples=int(1500/batch_size),callbacks=callbacks_list)


	loss = tmp_history.history['loss']
	val_loss = tmp_history.history['val_loss']
	

	K.clear_session()

	return val_loss[len(val_loss)-1]


## returns val_loss
def run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=50,drops=2,batch_size=32,lr=.001,plot_folder ='PLS_Out/',version=2,DA=True,callback=None,save_every=True,verbose=1,lower_learning_rate=False,lr_step=500,lr_drop_pc=.5,linked=False,num_stacks=1,tv_train=None,tv_test=None,g_tv_ssim=False):
    
    loss = []
    val_loss = []
    lrs = []
    
    
    if version==2:
        # checkpoint
        filepath=plot_folder + model.name + "/weights-improvement-{epoch:02d}-{loss:.2f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
        callbacks_list = []
        generate_folder(plot_folder + model.name)
        if save_every:
            callbacks_list.append(checkpoint)
        if callback is not None:
            callbacks_list.append(callback)
            
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=lr_drop_pc, epsilon=0.0001,
                  patience=50, min_lr=0.0000001,cooldown=10,verbose=verbose)
        
        if lower_learning_rate:
            callbacks_list.append(reduce_lr)
        
        # def step_decay(epoch):
#             lrate = lr#float(lr) * math.pow(lr_drop_pc, math.floor(1+float(epoch)/float(lr_step)))
#             return lrate
#         
#         if lower_learning_rate:
#             lrate = LearningRateScheduler(step_decay)
#             callbacks_list.append(lrate)
        
        
        #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.00001, patience=51, verbose=verbose)
        #callbacks_list.append(reduce_lr)
        #callbacks_list.append(early_stopping)
        # Fit the model
        
        if DA:
            datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=False,  # randomly flip images
            vertical_flip=False)  # randomly flip images
            
            # Compute quantities required for featurewise normalization
            # (std, mean, and principal components if ZCA whitening is applied).
            datagen.fit(X_train)

            # Fit the model on the batches generated by datagen.flow().
            
            tmp_history = model.fit_generator(datagen.flow(X_train, Y_train,
                                     batch_size=batch_size),
                        samples_per_epoch=X_train.shape[0],
                        nb_epoch=nb_epoch,
                        validation_data=(X_test, Y_test),callbacks=callbacks_list)
                        
        else: # don't use data augmentation
            if linked: #multiple outputs, need to slightly change this to double the Y_train twice 
                Y_trains = []
                Y_tests = []
                for ii in range(num_stacks+1):
                    Y_trains.append(Y_train)
                    Y_tests.append(Y_test)
                tmp_history = model.fit(X_train, Y_trains, batch_size=batch_size, nb_epoch=nb_epoch,
                      verbose=verbose, validation_data=(X_test, Y_tests),shuffle=True,callbacks=callbacks_list)
            elif g_tv_ssim:
                tmp_history = model.fit([X_train, tv_train], Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      verbose=verbose, validation_data=([X_test,tv_test], Y_test),shuffle=True,callbacks=callbacks_list)

            else:
                tmp_history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
                      verbose=verbose, validation_data=(X_test, Y_test),shuffle=True,callbacks=callbacks_list)

        
        loss = tmp_history.history['loss']
        val_loss = tmp_history.history['val_loss']
        
    else:
        lr_drop = int(float(nb_epoch)/(drops+1))
        for i in range(nb_epoch):
            print('Epoch: ' + str(i) + '. Learning rate: ' + str(K.get_value(model.optimizer.lr)))
            tmp_history = model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=1,
                      verbose=verbose, validation_data=(X_test, Y_test),shuffle=True)
        
            loss.append(tmp_history.history['loss'][0])
            val_loss.append(tmp_history.history['val_loss'][0])
            lrs.append(K.get_value(model.optimizer.lr))
        
            if lr_drop !=0:
                if i%lr_drop ==0 and i>0:
                    K.set_value(model.optimizer.lr, 0.1 * K.get_value(model.optimizer.lr))
        

    
    
    # Plot and save to file
    # f_out = plot_folder + model.name + '/training_plot.png'
#     
#     generate_folder(plot_folder + model.name)
#     from matplotlib.backends.backend_agg import FigureCanvasAgg
#     from matplotlib.figure import Figure
#     fig = Figure(figsize=(8,5))
#     ax = fig.add_subplot(1,1,1)
#     ax.plot(loss)
#     ax.plot(val_loss)
#     ax.set_title('Model Loss over time')
#     ax.set_ylabel('Loss')
#     ax.set_xlabel('epoch')
#     ax.legend(['train', 'test'], loc='upper left')
#     
#     canvas = FigureCanvasAgg(fig)
#     canvas.print_figure(f_out, dpi=80)
    
    return val_loss[len(val_loss)-1]
    
def save_training_plot(loss,val_loss,lr,model_name,plot_folder='PLS_Out/',suffix=''):
    f_out = plot_folder + model_name + '/training_plot'+ suffix+ '.png'
    
    loss = np.asarray(loss)
    val_loss = np.asarray(val_loss)
    
    generate_folder(plot_folder + model_name)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(8,5))
    ax = fig.add_subplot(1,1,1)
    ax2 = ax.twinx()
    ax.plot(loss)
    ax.plot(val_loss)
    ax.set_title('Model Loss over time')
    ax.set_ylabel('Loss')
    
    if len(loss)>1:
        mmin = np.min((np.percentile(loss, 10),np.percentile(val_loss,10)))
        mmax = np.max((np.percentile(loss, 90),np.percentile(val_loss,90)))
        mmin -= abs(mmin-np.max((np.percentile(loss,50),np.percentile(val_loss,50))))
        mmax += abs(mmax-np.min((np.percentile(loss,50),np.percentile(val_loss,50))))
        ax.set_ylim([mmin,mmax])
    
    
    ax.set_xlabel('epoch')
    
    ax2.plot(lr,'r')
    ax2.set_ylabel('Learning Rate', color='r')
    
    ax.legend(['train', 'test'], loc='upper left')
    
    canvas = FigureCanvasAgg(fig)
    canvas.print_figure(f_out, dpi=80)
    
    
        

def save_model(model,name,folder='models/',weights=True):
    model_name = folder + name
    model_yaml = model.to_yaml()
    if weights:
        model.save_weights(model_name+".h5")
    
    print("Saved model to disk : " +model_name)
    with open(model_name+ ".yaml", "w") as yaml_file:
        yaml_file.write(model_yaml)

def create_model_image_prediction():
    kernel_size = (9,9)
    
    model = Sequential()
    
    model.add(Convolution2D(64, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape,name = 'conv1'))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 1, 1,name ='conv2'))
    model.add(Activation('relu'))
    model.add(Convolution2D(1, kernel_size[0], kernel_size[1],name ='conv3'))
    model.add(Activation('relu'))
    
    return model


def load_H_g_target(version=2,index=1,theta=None,dirname=None,H_DIRNAME=None):
    import xray.recon.io
    
    if H_DIRNAME is None:
        H_DIRNAME='../xct-parallelbeam-matlab/system-matrix/'
    
    if dirname is None:
        DIRNAME = '/home/dlshare/xray-limitedview/samples/v'+str(version)+'/SamplesCombined/'
    else:
        DIRNAME=dirname

    if dirname is None:
        if theta is not None:
            f_true = xray.recon.io.read_true_image(DIRNAME,index,version)
            f_recon = xray.recon.io.read_recon_image(DIRNAME,index,theta=theta,version=3)
            g = xray.recon.io.read_meas_data(DIRNAME,index,theta=theta,version=3)
            H = xray.recon.io.read_system_matrix(H_DIRNAME,outtype='np',version=2,theta=theta)
            return H,g,f_true,f_recon
    
        if version==2:
            # Load H, g, and target fs
            H = xray.recon.io.read_system_matrix(H_DIRNAME,outtype='np',version=version)
            g = xray.recon.io.read_meas_data(DIRNAME, index)
            f_true = xray.recon.io.read_true_image(DIRNAME,index)
            f_recon = xray.recon.io.read_recon_image(DIRNAME,index)
    else:
        if theta is not None:
            H = xray.recon.io.read_system_matrix(H_DIRNAME,outtype='np',version=3,theta=theta)
            DIRNAME = dirname
            g = xray.recon.io.read_meas_data(DIRNAME, index)
            f_true = xray.recon.io.read_true_image(DIRNAME,index)
            f_recon = xray.recon.io.read_recon_image(DIRNAME,index)
        else:
            H = xray.recon.io.read_system_matrix(H_DIRNAME,outtype='np',version=version)
            DIRNAME = dirname
            g = xray.recon.io.read_meas_data(DIRNAME, index)
            f_true = xray.recon.io.read_true_image(DIRNAME,index)
            f_recon = xray.recon.io.read_recon_image(DIRNAME,index)
    
    # MNIST!
    if version==6:
        from keras.datasets import mnist
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        X_train = (X_train.astype(np.float32) - 127.5)/127.5
        H = None
        g = None
        f_true = X_train[index,:,:]
        f_recon = X_train[index,:,:]
    
    if len(f_true)==4*len(f_recon):
        from scipy.misc import imresize
        f_true = imresize(f_true.reshape(512,512), .5, mode='F').reshape(256*256,)
    
    return H,g,f_true,f_recon


def generate_folder(folder):
    import os
    if not os.path.exists(folder):
        os.makedirs(folder)


def PLS(model_name,version=2,index=8018,display=True,output_folder=None,weight_file=None,short=False,short_index='',ratio=100,data_creation=False,lamb=10,deploy=True,img_shape=(256,256,1),normalize=1,randomize_start=False):

    if output_folder is not None:
        import os
        if not os.path.exists('PLS_Out/' +output_folder + '/'):
            os.makedirs('PLS_Out/' + output_folder + '/')
  
    # Load H, g, and target fs
    H,g,f_true,f_recon = load_H_g_target(version=version,index=index)    
    
    if display is False:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(17,8))
    else:
        from matplotlib import pylab as pl
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        pl.close('all')
        fig = pl.figure(figsize=(17,8))
        pl.ion()
        
    
    
    def setup_subplot(fig,im,title,subplot):
        ax = fig.add_subplot(2,3,subplot)
        cax = ax.imshow(im, interpolation='nearest')
        ax.set_title(title)
        fig.colorbar(cax)

    def subplot_profile(fig,x,true_y,recon_y,deep_y,subplot=3):
        ax = fig.add_subplot(2,3,subplot)
        ax.plot(x,true_y)
        ax.plot(x,recon_y)
        ax.plot(x,deep_y)
        ax.legend(['True Image profile', 'Recon Image profile', 'Current Iteration profile'], loc='upper left',prop={'size':6})
        ax.set_title('Profile Comparison -- horizontal line at y=128')
   
    
    
    def save_plot(name,save_folder='PLS_Out/'+output_folder +'/'+short_index):
        canvas = FigureCanvasAgg(fig)
        canvas.print_figure(save_folder+ name +'.png', dpi=80)
    
    
    def run_PLS(model_name=model_name,include_df=True,niters=300,lr=1,lamb=lamb,save_iterations=np.array([4,49,149,299]),weight_file=weight_file,ratio=ratio,deploy=deploy,img_shape=img_shape,normalize=normalize,randomize_start=randomize_start):
        
        ratio = lamb # talk about confusing... no longer using ratio but don't wanna change older code which does use ratio
        
        K.clear_session()
        K.set_learning_phase(0)
        model = load_trained_CNN(name=model_name,deploy=deploy,weight_file=weight_file)
        
        input_img = model.input
        layer_name = model.layers[len(model.layers)-1].name
        
        num_dim = img_shape[0]*img_shape[1]*img_shape[2]
        img_dim = int(np.sqrt(num_dim))
        
        x = list(range(1,img_dim+1))
        y_vert = int(img_dim/2)
        
        
        
        if include_df:
            plot_name = 'df_dR_'
            f_deep = np.zeros(num_dim,)
        else:
            plot_name = 'dR_'
            f_deep = f_recon.reshape(num_dim,)
            
        if randomize_start:
            f_deep = f_recon
#             f_deep[17:28,:]=-1
            f_deep = f_deep.reshape(num_dim,)
        
        print_len = 10
        print_t = time.time()
        
        
        for i in range(niters):
            if i%print_len ==0:
                print('On iteration ' + str(i) + ', took ' +str(time.time()-print_t) + ' s.')
                print_t = time.time()
        
            # Normalize and get CNN derivative
            if normalize==1:
                mmin = np.min(f_deep)
                mmax = np.max(f_deep)
                norm_deep = (f_deep.reshape(1,img_shape[0],img_shape[1],img_shape[2])-mmin)/(mmax-mmin)
            
            if normalize==2:
                norm_deep = f_deep.reshape(1,img_shape[0],img_shape[1],img_shape[2])
                
            t = time.time()
            df_CNN2 = compute_derivative(model,norm_deep,input_img,0,layer_name=layer_name)
            compute_time = time.time()-t
            
            # Non positivity constraint for CNN derivative -- thye arn't gonna like this
            #high_values_indices = df_CNN2[1] >0  # Where values are low
            #df_CNN2[1][high_values_indices] = 0
            
            # If we are including the derivative from data fidelity term
            if include_df:
                df_deep = np.transpose(H)*(H*f_deep - g)
                f_deep = f_deep - lr*(df_deep)
                # Set lamb
                mag_df = np.linalg.norm(df_deep)
                mag_df_CNN2 = np.linalg.norm(df_CNN2[1])
                lamb = ((mag_df/mag_df_CNN2)/ratio)
                lamb=ratio
                
            
            if not (i==0):
                f_deep = f_deep + lamb*lr*df_CNN2[1].reshape(num_dim,)
            
            # Non negativity constraint
            if normalize==1:
                low_values_indices = f_deep < 0  # Where values are low
                f_deep[low_values_indices] = 0
            
            # Display magnitude and time taken to compute derivative of CNN
            if display and include_df:
                print('Took ' + str(compute_time) + ' s to compute df_CNN2. ')
                mag_df = np.linalg.norm(df_deep)
                mag_df_CNN2 = np.linalg.norm(df_CNN2[1])
                print('Magnitude of df_deep: ' + str(mag_df) + '. Mag of df_CNN2: ' + str(mag_df_CNN2) + '. Ratio: ' + str(mag_df/mag_df_CNN2) +  '. Lambda: ' + str(lamb))
                
            # Plot stuff, either to display or to save out
            if display or i in save_iterations:
                fig.clf()
                setup_subplot(fig,f_true.reshape(img_dim,img_dim),'True Image. Id:' + str(index),1)
                setup_subplot(fig,f_recon.reshape(img_dim,img_dim),'Recon Image.',2)
                setup_subplot(fig,f_deep.reshape(img_dim,img_dim),'Current Reconstruction. Iteration: ' + str(i),4)
                
                if include_df:
                    setup_subplot(fig,(df_deep.reshape(img_dim,img_dim)*-1)*lr,'df_data fidelity. Mag: ' +'{0:.2f}'.format(np.linalg.norm(df_deep)),6)
                
                prediction = model.predict(f_deep.reshape(1,img_shape[0],img_shape[1],img_shape[2]))
                setup_subplot(fig,(df_CNN2[1].reshape(img_dim,img_dim)*-1)*lr*lamb,'df_CNN. Prediction: ' + '{0:.3f}'.format(prediction[0][0]) + '. Loss : '  +'{0:.7f}'.format(df_CNN2[0]),5)
                true_y = np.squeeze(f_true.reshape(img_dim,img_dim)[y_vert,:])
                recon_y = np.squeeze(f_recon.reshape(img_dim,img_dim)[y_vert,:])
                deep_y = np.squeeze(f_deep.reshape(img_dim,img_dim)[y_vert,:])
                subplot_profile(fig,x,true_y=true_y,recon_y=recon_y,deep_y=deep_y,subplot=3)
       
            if display:
                pl.show()
                pl.pause(.01)
        
            if i in save_iterations:
                if output_folder is not None:
                    save_plot(plot_name + str(i))
            
            # Theres some kind of GPU memory leak here.  Let's resolve it!
            if compute_time>.8:
                K.clear_session()
                K.set_learning_phase(0)
                model = load_trained_CNN(name=model_name,deploy=deploy,weight_file=weight_file)
                input_img = model.input
                
        return f_deep

    if data_creation:
        data = run_PLS(include_df=True,niters=200,save_iterations=np.arrya([199]))
        return data
    if short:
        tmp = run_PLS(include_df=False,niters=20,save_iterations=np.array([1]))
    else:
        # run PLS for only the regularization term
        if not display:
            tmp = run_PLS(include_df=False,niters=100,save_iterations=np.array([1,4,9,14,19,24,29,49,99]))

        # run PLS for both data fidelity term and regularization term
        tmp = run_PLS(include_df=True,niters=300,save_iterations=np.array([1,4,9,14,19,24,29,49,149,299]))
    
    
def project_img(model,img,linked=False):
    in_shape = model.input_shape
    if linked:
        out_shape = model.output_shape[0]
    else:
        out_shape = model.output_shape
    
    total_len = img.shape[0]
    entire_img = np.zeros(img.shape)
    entire_img[:,:] = np.min(img)
    sy, ey = 0,in_shape[1]
    dif = int((in_shape[1]-out_shape[1])/2)
    count = 0
    while(ey <= total_len):
        sx,ex = 0, in_shape[1]
        while(ex <= total_len):
            prediction = model.predict(img[sx:ex,sy:ey].reshape((1,)+in_shape[1:]), verbose=0)
            if linked: prediction = prediction[0]      
            entire_img[sx+dif:ex-dif,sy+dif:ey-dif] = prediction.reshape(prediction.shape[1:3])
            ex+=out_shape[1]
            sx+=out_shape[1]
            count+=1
        ey+=out_shape[1]
        sy+=out_shape[1]

    if count > 1:
        print('Count is : ' + str(count) + ', when it hsould be just 1')
    
    return entire_img
 
def Projected_PLS(model,njumps=5,version=2,theta=None,index=8018,display=True,output_folder=None,short_index='',img_shape=(256,256,1),data_dirname=None,niters=2500,cutoff=.0075,H_DIRNAME=None,linked=False,Verbose=1,TVConstraint=False,tau=838):

    
    if output_folder is not None:
        import os
        if not os.path.exists('PLS_Out/' +output_folder + '/'):
            os.makedirs('PLS_Out/' + output_folder + '/')
  
    # Load H, g, and target fs
    H,g,f_true,f_recon = load_H_g_target(version=version,index=index,theta=theta,dirname=data_dirname,H_DIRNAME=H_DIRNAME)    
    
    if display is False:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(17,8))
    else:
        from matplotlib import pylab as pl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        pl.close('all')
        fig = pl.figure(figsize=(17,8))
        pl.ion()
        
    
    
    def setup_subplot(fig,im,title,subplot):
        ax = fig.add_subplot(2,3,subplot)
        cax = ax.imshow(im, interpolation='nearest')
        ax.set_title(title)
        fig.colorbar(cax)

    def subplot_profile(fig,x,true_y,recon_y,deep_y,subplot=3):
        ax = fig.add_subplot(2,3,subplot)
        ax.plot(x,true_y)
        ax.plot(x,recon_y)
        ax.plot(x,deep_y)
        ax.legend(['True Image profile', 'Recon Image profile', 'Current Iteration profile'], loc='upper left',prop={'size':6})
        ax.set_title('Profile Comparison -- horizontal line at y=128')
   
    
    if output_folder is not None:
        def save_plot(name,save_folder='PLS_Out/'+output_folder +'/'+short_index):
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(save_folder+ name +'.png', dpi=80)
    
    
    
    def run_PLS(model=model,niters=niters,lr=.75,save_iterations=[],img_shape=img_shape,display=display,Verbose=Verbose,g=g,H=H,f_true=f_true,f_recon=f_recon):
            
        display_len = 10
        normalize=0
        num_dim = img_shape[0]*img_shape[1]*img_shape[2]
        img_dim = int(np.sqrt(num_dim))
        
        x = list(range(1,img_dim+1))
        y_vert = int(img_dim/2)
        
        
        prior_dont_project=niters/10
        plot_name ='df_p'
        f_deep = f_recon.reshape(num_dim,)
        
        print_len = 100
        print_t = time.time()
        
        psnr_save =[]
        projection_times_save = []
        magnitudes_save = []
        data_fidelity_loss = []
        peak_saves = []
        peak_saves_iterations = []
        psnr_save_actual = []
        
        outer_padding = 0
        mask = np.zeros((img_shape[0],img_shape[1]))
        mask[outer_padding:img_shape[0]-outer_padding,outer_padding:img_shape[1]-outer_padding] = 1
        
#         num_regular_projected = 1000
#         cutoff 
        
        num_projections = 0
        
        #for i in range(niters):
        i=-1
        nj=0
#         njumps=15
        
        cost = .5*np.sum(np.power(H*f_deep - g,2))
        data_fidelity_loss.append(cost)
        
        
#         stopping_rule_loss = .5*np.sum(np.power(H*f_deep - g,2))
#         print('Stopping rule loss: ' + str(stopping_rule_loss))
        
        while nj <= njumps and i < niters:
            i+=1
            psnr_save.append(MSE(f_deep.reshape(256*256,),f_true))#,pixel_max=1,pixel_min=0))
            
            
            if (i-1)%print_len ==0 and Verbose>0:
                print('On iteration ' + str(i) + '. mse: ' + str(psnr_save[-1]) +  ', cost: ' + str(cost) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
                print_t = time.time()
            
            f_deep = np.reshape(f_deep,(num_dim,))
            
            
            
            df_deep = np.transpose(H)*(H*f_deep - g)
            f_deep = f_deep - lr*(df_deep)
            
            
            
            magnitudes_save.append(np.linalg.norm(df_deep))
            
            cost = .5*np.sum(np.power(H*f_deep - g,2))
            data_fidelity_loss.append(cost)
            
            if i==0:
                stopping_rule_loss = cost
                if Verbose>0:
                    print('Stopping rule loss: ' + str(stopping_rule_loss))
                

            
            # nonneg
            low_values_indices = f_deep < 0  # Where values are low
            f_deep[low_values_indices] = 0
            
            if tv_constraint:
                f_deep = perform_tv_projection(np.reshape(f_deep,(256,256)),tau=tau)
                f_deep = np.reshape(f_deep,(num_dim,))
            
            
            # Make outer edge 0
#             f_deep *= mask.reshape((len(f_deep),))
#             if i%num_regular_projected ==0:
            norm_deep = f_deep
            rel_cost_diff = (data_fidelity_loss[i] - data_fidelity_loss[i+1])/data_fidelity_loss[i+1]
            if i==0 or (rel_cost_diff < cutoff and rel_cost_diff>0): #(cost <= stopping_rule_loss):# 
                if Verbose>0:
                    print('projecting!')
                    print('On iteration ' + str(i) + '. mse: ' + str(MSE(f_deep.reshape(256*256,),f_true)) +  ', rel_cost_diff: ' + str(rel_cost_diff) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
                
                nj+=1
                save_iterations.append(i)

                projection_times_save.append(1)
                # Normalize and get CNN derivative
                norm_deep = f_deep
                
            
                norm_deep = norm_deep.reshape((256,256))
                
                peak_saves.append(np.copy(norm_deep))
                peak_saves_iterations.append(i)
#                 t = time.time()
#                 df_CNN2 = project_img(model,norm_deep,linked=linked)
                
#                 compute_time = time.time()-t
                
                
                
                norm_deep = norm_deep.reshape((1,256,256,1))
                
                if i>0:
                    old_pred = pred
                
                pred = model.predict(norm_deep)
                
                if i>0:
                    mse_change_projections = MSE(np.squeeze(pred),np.squeeze(old_pred))
                    if Verbose>0:
                        print('Change in mse between projections is : ' + str(mse_change_projections))
                
                md = 0
                dif = int((model.input_shape[2]-model.output_shape[2])/2)
                ts = model.input_shape[2]
                ts2 = model.output_shape[2]
        
                psnr_projection = MSE(f_true.reshape(1,ts,ts,1)[:,dif:ts-dif,dif:ts-dif,:],pred[:,md:ts2-md,md:ts2-md,:])
                if Verbose>0:
                    print('mse, projection: ' + str(psnr_projection));
             
                norm_deep[:,dif:ts-dif,dif:ts-dif,:] = pred
                norm_deep.reshape((256,256))
                
                peak_saves.append(np.copy(norm_deep))
                peak_saves_iterations.append(i+1)
                
                f_deep = norm_deep
                num_projections += 1
                cost = .5*np.sum(np.power(H*np.reshape(f_deep,(num_dim,)) - g,2))
                rel_cost_diff = (data_fidelity_loss[i+1] - cost)/cost
                if Verbose>0:
                    print('Post projection ' + str(num_projections) + ', . mse: ' + str(MSE(f_deep.reshape(256*256,),f_true)) +  ', cost: ' + str(cost) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
                
                
            else:
                projection_times_save.append(0)
                
            #print(psnr(f_deep.reshape(256*256,),f_true,pixel_max=1,pixel_min=0))
            # Display magnitude and time taken to compute derivative of CNN
                
            # Plot stuff, either to display or to save out
            if (display and i%display_len==0) or i in save_iterations:
                fig.clf()
                setup_subplot(fig,f_true.reshape(img_dim,img_dim),'True Image. Id:' + str(index),1)
                setup_subplot(fig,f_recon.reshape(img_dim,img_dim),'Recon Image.',2)
                setup_subplot(fig,f_deep.reshape(img_dim,img_dim),'Current Reconstruction. Iteration: ' + str(i),4)
                
#                 setup_subplot(fig,(norm_deep.reshape(img_dim,img_dim)*-1),'Input to Projection CNN',6)
                
                setup_subplot(fig,df_deep.reshape(img_dim,img_dim),'data fidelity derivative',5)
                
                true_y = np.squeeze(f_true.reshape(img_dim,img_dim)[y_vert,:])
                recon_y = np.squeeze(f_recon.reshape(img_dim,img_dim)[y_vert,:])
                deep_y = np.squeeze(f_deep.reshape(img_dim,img_dim)[y_vert,:])
                subplot_profile(fig,x,true_y=true_y,recon_y=recon_y,deep_y=deep_y,subplot=3)
                
                ax = fig.add_subplot(2,3,6)
                ax2 = ax.twinx()
                ax.plot(psnr_save)
                ax.set_title('PSNR over LS optimization')
                ax.set_ylabel('PSNR')
                ax.set_xlabel('iteration')

                ax2.plot(magnitudes_save,'r')
                ax2.set_ylabel('Magnitude of data fidelity update',color='r')
                ax2.set_ylim([0,.05])
                   
            if display:
                pl.show()
                pl.pause(.01)
                #print('Compute time:' + str(compute_time))
                
        
            if i in save_iterations:
                if output_folder is not None:
                    save_plot(plot_name + str(i))
            
                
        return f_deep,psnr_save,projection_times_save,magnitudes_save,peak_saves,peak_saves_iterations,data_fidelity_loss

    return run_PLS()
    
    
def Projected_PLS_MP_PI(model,njumps=5,version=2,theta=None,index=8018,display=True,output_folder=None,short_index='',img_shape=(256,256,1),data_dirname=None,niters=2500,cutoff=.0075,H_DIRNAME=None,linked=False,MP=None,inv_MP=None,Verbose=1):

    if MP is None:
        print ('wtf... MP matrix is none, we need a matrix here, 256^2 x 256^2')
        
    if output_folder is not None:
        import os
        if not os.path.exists('PLS_Out/' +output_folder + '/'):
            os.makedirs('PLS_Out/' + output_folder + '/')
  
    # Load H, g, and target fs
    H,g,f_true,f_recon = load_H_g_target(version=version,index=index,theta=theta,dirname=data_dirname,H_DIRNAME=H_DIRNAME)    
    
    if display is False:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        fig = Figure(figsize=(17,8))
    else:
        print('dispplaying, I promise')
        import matplotlib
        matplotlib.use('TkAgg')
        from matplotlib import pylab as pl
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        pl.close('all')
        fig = pl.figure(figsize=(17,8))
        pl.ion()
        
    
    
    def setup_subplot(fig,im,title,subplot):
        ax = fig.add_subplot(2,3,subplot)
        cax = ax.imshow(im, interpolation='nearest')
        ax.set_title(title)
        fig.colorbar(cax)

    def subplot_profile(fig,x,true_y,recon_y,deep_y,subplot=3):
        ax = fig.add_subplot(2,3,subplot)
        ax.plot(x,true_y)
        ax.plot(x,recon_y)
        ax.plot(x,deep_y)
        ax.legend(['True Image profile', 'Recon Image profile', 'Current Iteration profile'], loc='upper left',prop={'size':6})
        ax.set_title('Profile Comparison -- horizontal line at y=128')
   
    
    if output_folder is not None:
        def save_plot(name,save_folder='PLS_Out/'+output_folder +'/'+short_index):
            canvas = FigureCanvasAgg(fig)
            canvas.print_figure(save_folder+ name +'.png', dpi=80)
    
    
    
    def run_PLS(model=model,niters=niters,lr=.75,save_iterations=[300,301,600,601, 3000, 3001, 6000,6001],img_shape=img_shape,display=display,Verbose=Verbose):
        
        display_len = 10
        normalize=0
        num_dim = img_shape[0]*img_shape[1]*img_shape[2]
        img_dim = int(np.sqrt(num_dim))
        
        x = list(range(1,img_dim+1))
        y_vert = int(img_dim/2)
        
        
        prior_dont_project=niters/10
        plot_name ='df_p'
        f_deep = f_recon.reshape(num_dim,)
        
        print_len = 100
        print_t = time.time()
        
        psnr_save =[]
        projection_times_save = []
        magnitudes_save = []
        data_fidelity_loss = []
        peak_saves = []
        peak_saves_iterations = []
        psnr_save_actual = []
        
        outer_padding = 10
        mask = np.zeros((img_shape[0],img_shape[1]))
        mask[outer_padding:img_shape[0]-outer_padding,outer_padding:img_shape[1]-outer_padding] = 1
        
#         num_regular_projected = 1000
#         cutoff 
        
        num_projections = 0
        
        #inv_MP = 1-MP
        tic2 = time.time()
        f_measurable_component = np.dot(MP,f_true)
        toc2 = time.time()
        print('Took ' +str(toc2-tic2) + ' s to compute null component.')
        
        #for i in range(niters):
        i=-1
        nj=0
#         njumps=15
        
        cost = .5*np.sum(np.power(H*f_deep - g,2))
        data_fidelity_loss.append(cost)
        
        
#         stopping_rule_loss = .5*np.sum(np.power(H*f_deep - g,2))
#         print('Stopping rule loss: ' + str(stopping_rule_loss))
        
        while nj <= njumps:
            i+=1
            psnr_save.append(MSE(f_deep.reshape(256*256,),f_true))#,pixel_max=1,pixel_min=0))
            
            
            if (i-1)%print_len ==0 and Verbose>0:
                print('On iteration ' + str(i) + '. mse: ' + str(psnr_save[-1]) +  ', cost: ' + str(cost) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
                print_t = time.time()
            
            f_deep = np.reshape(f_deep,(num_dim,))
            
            
            
#             df_deep = np.transpose(H)*(H*f_deep - g)
            #f_deep = f_deep - lr*(df_deep)
            if nj > 0:
                tic2 = time.time()
                f_deep = f_measurable_component + np.dot((inv_MP),f_deep)
                toc2 = time.time()
                if Verbose>0:
                    print('Took ' +str(toc2-tic2) + ' s to compute null component.')
#             magnitudes_save.append(np.linalg.norm(df_deep))
            

            cost = .5*np.sum(np.power(H*f_deep - g,2))
            data_fidelity_loss.append(cost)
            
            norm_deep = f_deep
            rel_cost_diff = (data_fidelity_loss[i] - data_fidelity_loss[i+1])/data_fidelity_loss[i+1]
            if Verbose>0:
                print('projecting!')
                print('On iteration ' + str(i) + '. mse: ' + str(MSE(f_deep.reshape(256*256,),f_true)) +  ', rel_cost_diff: ' + str(rel_cost_diff) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
            
            nj+=1
            save_iterations.append(i)

            projection_times_save.append(1)
        
            norm_deep = norm_deep.reshape((256,256))
            
            peak_saves.append(np.copy(norm_deep))
            peak_saves_iterations.append(i)
            norm_deep = norm_deep.reshape((1,256,256,1))
            
            pred = model.predict(norm_deep)
            md = 0
            dif = int((model.input_shape[2]-model.output_shape[2])/2)
            ts = model.input_shape[2]
            ts2 = model.output_shape[2]
    
            psnr_projection = MSE(f_true.reshape(1,ts,ts,1)[:,dif:ts-dif,dif:ts-dif,:],pred[:,md:ts2-md,md:ts2-md,:])
            if Verbose>0:
                print('mse, projection: ' + str(psnr_projection));
             
            norm_deep[:,dif:ts-dif,dif:ts-dif,:] = pred
            norm_deep.reshape((256,256))
            
            peak_saves.append(np.copy(norm_deep))
            peak_saves_iterations.append(i+1)
            
            f_deep = norm_deep
            num_projections += 1
            cost = .5*np.sum(np.power(H*np.reshape(f_deep,(num_dim,)) - g,2))
            rel_cost_diff = (data_fidelity_loss[i+1] - cost)/cost
        
            if Verbose>0:
                print('Post projection ' + str(num_projections) + ', . mse: ' + str(MSE(f_deep.reshape(256*256,),f_true)) +  ', cost: ' + str(cost) + ',PSNR:' + str(psnr(f_deep.reshape(256*256,),f_true,1,0)))
            
            data_fidelity_loss.append(cost)
                
            #print(psnr(f_deep.reshape(256*256,),f_true,pixel_max=1,pixel_min=0))
            # Display magnitude and time taken to compute derivative of CNN
                
            # Plot stuff, either to display or to save out
            if (display) or i in save_iterations:
                fig.clf()
                setup_subplot(fig,f_true.reshape(img_dim,img_dim),'True Image. Id:' + str(index),1)
                setup_subplot(fig,f_recon.reshape(img_dim,img_dim),'Recon Image.',2)
                setup_subplot(fig,f_deep.reshape(img_dim,img_dim),'Current Reconstruction. Iteration: ' + str(i),4)
                
#                 setup_subplot(fig,(norm_deep.reshape(img_dim,img_dim)*-1),'Input to Projection CNN',6)
                
                #setup_subplot(fig,df_deep.reshape(img_dim,img_dim),'data fidelity derivative',5)
                
                true_y = np.squeeze(f_true.reshape(img_dim,img_dim)[y_vert,:])
                recon_y = np.squeeze(f_recon.reshape(img_dim,img_dim)[y_vert,:])
                deep_y = np.squeeze(f_deep.reshape(img_dim,img_dim)[y_vert,:])
                subplot_profile(fig,x,true_y=true_y,recon_y=recon_y,deep_y=deep_y,subplot=3)
                
                ax = fig.add_subplot(2,3,6)
                ax2 = ax.twinx()
                ax.plot(psnr_save)
                ax.set_title('PSNR over LS optimization')
                ax.set_ylabel('PSNR')
                ax.set_xlabel('iteration')

                ax2.plot(magnitudes_save,'r')
                ax2.set_ylabel('Magnitude of data fidelity update',color='r')
                ax2.set_ylim([0,.05])
                       
            if display:
                pl.show()
                pl.pause(.01)
                #print('Compute time:' + str(compute_time))
                
        
            if i in save_iterations:
                if output_folder is not None:
                    save_plot(plot_name + str(i))
            
                
        return f_deep,psnr_save,projection_times_save,magnitudes_save,peak_saves,peak_saves_iterations,data_fidelity_loss

    return run_PLS()


def create_AD_dataset_IC(model,njumps=5,version=3,theta=None,img_shape=(256,256,1),data_dirname=None,MP=None,inv_MP=None,output_directory='tmp/',max_index=7500,min_index=0):
    
    dif = int((model.input_shape[2]-model.output_shape[2])/2)
    ts = model.input_shape[2]
    tic = time.time()
    for i in range(max_index-min_index):
        index = i+min_index
        print('On index:' + str(index) + ' in ' + str(time.time()-tic) + ' s')
        tic=time.time()
        
        X_train = []
        Y_train = []
        save_file = output_directory + str(index) + '.pkl'
        
        # Load dataset
        H,g,f_true,f_recon = load_H_g_target(version=version,index=index,theta=theta,dirname=data_dirname,H_DIRNAME=None)    
    
        # Calculate measurable component
        f_measurable_component = np.dot(MP,f_true)
        f_deep = f_measurable_component.copy()
    
        #while loop - predict, save prediction, add true meas component and null component of prediction, save that
        num_projections = 0
        while num_projections < njumps:
            pred = model.predict(f_deep.reshape((1,256,256,1)))
            f_deep = f_measurable_component.copy().reshape((1,256,256,1))
            f_deep[:,dif:ts-dif,dif:ts-dif,:] = pred
            
            # save_prediction
            X_train.append(f_deep.copy().reshape(img_shape))
            Y_train.append(f_true.reshape(img_shape))
            
            # add true meas and null component
            f_deep = f_measurable_component+np.dot(inv_MP,f_deep.reshape(f_measurable_component.shape))
            X_train.append(f_deep.copy().reshape(img_shape))
            Y_train.append(f_true.reshape(img_shape))
            
            num_projections+=1
    
        data = {}
        data['X_train'],data['y_train'] = X_train,Y_train
        with open(save_file, 'wb') as f:
            pickle.dump(data,f)
            
        
    
    
    



def psnr(img1, img2,pixel_max=1,pixel_min=-1):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0:
        return 100
    pixel_range = pixel_max-pixel_min
    return 20 * math.log10(pixel_range / math.sqrt(mse))


def ssim(img1,img2):
    from skimage.measure import compare_ssim as ssim
    
    return ssim(img1, img2)
    
def MSE(img1,img2):
    return np.mean( (img1 - img2) ** 2 )
    

def compare_psnr(model2,njumps=5,mindex=1,cutoff=.0075,niters=2500,index=3,mf='',theta=None,display=True,PSNR=True,version=2,pls=True,data_dirname=None,title='tmp',linked=False,Verbose=1):
# 
#     try:
#         fname = 'PLS_Out/'+mf[19:-1] +'/training_plot.png'
#         training_plot = np.asarray(Image.open(fname))
#     except Exception:
#         training_plot = np.zeros((256,256))
#         #training_plot = np.asarray(Image.open(mf+'recon_input.png'))

# 
#     normalize = False
    H,g,f_true,f_recon = load_H_g_target(version=version,index=index,theta=theta,dirname=data_dirname)
#     md = 10+40
#     ts = 256
#     if PSNR:
#         psnr_recon = MSE(f_true.reshape(ts,ts)[md:ts-md,md:ts-md], f_recon.reshape(ts,ts)[md:ts-md,md:ts-md])
#         #psnr_recon = psnr(f_true.reshape(ts,ts), f_recon.reshape(ts,ts),pixel_max=1,pixel_min=0)
#     else:
#         psnr_recon = ssim(f_true.reshape(ts,ts)[md:ts-md,md:ts-md], f_recon.reshape(ts,ts)[md:ts-md,md:ts-md])
#     # normalize for CNN
#     if normalize:
#         f_true_orig = f_true
#         f_recon_orig = f_recon
#         f_true = normalize_single(f_true)
#         f_recon = normalize_single(f_recon)
#     else:
#         f_true_orig=f_true
#         f_recon_orig=f_recon
#         
#     md = 10
#     # Predict
#     pred = model2.predict(f_recon.reshape(1,256,256,1))
#     if linked:
#         pred = pred[0]
# 
#     # calc cropped psnr
#     if linked:
#         dif = int((model2.input_shape[2]-model2.output_shape[0][2])/2) + md
#         ts = model2.input_shape[2]
#         ts2 = model2.output_shape[0][2]
#     else:
#         dif = int((model2.input_shape[2]-model2.output_shape[2])/2) + md
#         ts = model2.input_shape[2]
#         ts2 = model2.output_shape[2]
#     
#     if PSNR:
#         psnr_projection = MSE(f_true.reshape(1,ts,ts,1)[:,dif:ts-dif,dif:ts-dif,:],pred[:,md:ts2-md,md:ts2-md,:])
#     else:
#         psnr_projection = ssim(f_true.reshape(ts,ts)[dif:ts-dif,dif:ts-dif],pred.reshape(ts2,ts2)[md:ts2-md,md:ts2-md])
# 
#     print('psnr_recon: ' + str(psnr_recon))
#     print('psnr_projection: ' + str(psnr_projection))




#     if pls:
    ls_starting_with_projection,psnr_save,projection_times_save,magnitudes_save,peak_saves,peak_saves_iterations,data_fidelity_loss = Projected_PLS(model2,njumps=njumps,niters=niters,cutoff=cutoff,version=version,theta=theta,index=index,display=display,data_dirname=data_dirname,linked=linked,Verbose=Verbose)

#     if PSNR:
#         psnr_ls = MSE(f_true_orig.reshape(ts,ts)[md:ts-md,md:ts-md], ls_starting_with_projection.reshape(ts,ts)[md:ts-md,md:ts-md])
#     else:
#         psnr_ls = ssim((f_true_orig).astype(float).reshape(ts,ts)[md:ts-md,md:ts-md], (ls_starting_with_projection).astype(float).reshape(ts,ts)[md:ts-md,md:ts-md])
    # else:
#         psnr_ls = 0
#         ls_starting_with_projection = np.zeros((256,256))

#     print('psnr_recon: ' + str(psnr_recon))
#     print('psnr_projection: ' + str(psnr_projection))
#     print('psnr_LS_projection:' + str(psnr_ls))

    #display is True:
    #from matplotlib.backends.backend_agg import FigureCanvasAgg
    #from matplotlib.figure import Figure
    # pl.close('all')
    # fig = pl.figure(figsize=(18,8))
    #fig = Figure(figsize=(16,12))

    # def setup_subplot(im,title,subplot,fig=fig,cb=True):
#         ax = fig.add_subplot(2,3,subplot)
#         cax = ax.imshow(im, interpolation='nearest')
#         ax.set_title(title)
#         if cb:fig.colorbar(cax)
# 
#     if pls:
#         def plot_psnr_plot():
#             ax = fig.add_subplot(2,3,6)
#             ax2 = ax.twinx()
#             ax.plot(psnr_save)
#             ax.set_title('PSNR over LS optimization')
#             ax.set_ylabel('PSNR')
#             ax.set_xlabel('iteration')
# 
#             ax2.plot(magnitudes_save,'r')
#             ax2.set_ylabel('Magnitude of data fidelity update',color='r')
#             ax2.set_ylim([0,.05])
#             # ax2.plot(projection_times_save,'r')
#             # ax2.set_ylabel('Projections via CNN', color='r')
# 
#     def plot_profile(imgs,strings,y_vert=130,img_dim=256):
#         x = list(range(1,img_dim))
#         ax = fig.add_subplot(2,3,3)
#         for i in range(len(strings)):
#             print('On : ' + strings[i] + ' .  dim: ' + str(imgs[i].shape))
#             ax.plot(np.squeeze(imgs[i].reshape(img_dim,img_dim)[y_vert,:]))
# 
#         ax.legend(strings, loc='upper left',prop={'size':6})
#         ax.set_title('Profile Comparison -- horizontal line at y=' + str(y_vert))
# 
# 
#     md2 = 30 # (55?)
#     def show_plot(psnr_ls=psnr_ls,ls_recon=ls_starting_with_projection.reshape(256,256)[md2:ts-md2,md2:ts-md2],ls_recon_uncropped=ls_starting_with_projection.reshape(256,256),psnr_recon=psnr_recon,psnr_projection=psnr_projection,model_output=pred.reshape(pred.shape[1],pred.shape[1]),tt=f_true_orig.reshape(256,256),ri=f_recon_orig.reshape(256,256),parsed_title='',training_plot=training_plot):
#         # pl.clf()
#         setup_subplot(ri,'Recon input.  SSIM:' + '{0:.5f}'.format(psnr_recon),2)
#         setup_subplot(tt,'True target',1)
#         # setup_subplot(training_plot,'Loss during Training',3,cb=False)
# 
#         setup_subplot(model_output,'Output from CNN.  SSIM:' + '{0:.5f}'.format(psnr_projection),4)
#         setup_subplot(ls_recon,'LS starting with projection. SSIM:  ' + '{0:.5f}'.format(psnr_ls),5)
#         if pls:
#             plot_psnr_plot()
# 
# 
#         imgs = []
#         imgs.append(ls_recon_uncropped)
#         imgs.append(ri)
#         imgs.append(tt)
#         #imgs.append(model_output)
#         strings = ['LS with CNN projecting','Recon input Image','True target Image']#,'Initial CNN Output']
#         plot_profile(imgs,strings)


    # pl.pause(.01)
    # pl.ion()
    #show_plot()
    #canvas = FigureCanvasAgg(fig)
    #canvas.print_figure(title + '.png', dpi=60)
    if pls:
        import scipy.io as sio
        matlab_transfer_folder = 'output_transfer_for_matlab/'
        fname = str(title) +'_'+ str(index) +'_'+ str(niters) +'_' + str(cutoff)
        trans = {}
        trans['g'] = g
        trans['H'] = H
        trans['f_true'] = f_true
        trans['f_recon'] = f_recon
        trans['ppls'] = ls_starting_with_projection
#         trans['proj_cnn'] = pred
        trans['psnr_save'] = psnr_save
        trans['peak_saves'] = np.asarray(peak_saves)
        trans['peak_saves_iterations'] = np.asarray(peak_saves_iterations)
        trans['magnitudes_save'] = magnitudes_save
        trans['data_fidelity_loss'] = data_fidelity_loss
        sio.savemat(matlab_transfer_folder + fname + '.mat',trans)





def make_dataset(trainf,testf,testFinalf,ntrain,ntest,ntestFinal,DIRNAME,AD_path=None,num_std_away=100):
#     import numpy as np
    import time
    import xray.recon.io
    import random
    from scipy.misc import imresize
#     import helper_functions as hf
#     import pickle

    generate_folder(trainf)
    generate_folder(testf)
    generate_folder(testFinalf)
    train_count = 0
    test_count = 0

    print(trainf)
    print(testf)
    print(testFinalf)
    print(DIRNAME)
    print(AD_path)
#     sys.exit()
    # load original recon/targs and toss into trainf/testf
    offset=0
    orig_shape = (256,256)
    targ_shape = (256-(offset*2),256-(offset*2))
    print_len=100
    Debugging = False
    
    def load_images_true(index=1,DIRNAME=DIRNAME):
        
        f_true = xray.recon.io.read_true_image(DIRNAME,index)
#         if Debugging: print('trying to load 2')
        f_recon = xray.recon.io.read_recon_image(DIRNAME,index)
        if Debugging: print('successful load')
        return f_true, f_recon

    # randomly crops the given true/recon, given offset and shapes above
    def random_crop(f_true,f_recon,offset=offset,orig_len=orig_shape[0],targ_len=targ_shape[0]):
        f_true_i = np.reshape(f_true,orig_shape)
        # f_true_i = imresize(f_true_i, .5, mode='F')
        f_recon_i = np.reshape(f_recon,orig_shape)
        rx = random.randint(offset,orig_len-(offset+targ_len))
        ry = random.randint(offset,orig_len-(offset+targ_len))
        crop_true = f_true_i[rx:rx+targ_len,ry:ry+targ_len]
        crop_recon = f_recon_i[rx:rx+targ_len,ry:ry+targ_len]
        if Debugging: print('successful crop')
        return crop_true,crop_recon

    def toss_into_folder(X,Y,fold,fname):
        data = {}
        data['X'],data['Y'] = X,Y
        with open(fold + fname + '.pkl', 'wb') as f:
            pickle.dump(data,f)
        
        if Debugging: print('successful toss')

    MSE_train = []
    # Original recon/targ images
    tic = time.time()
    for i in range(ntrain):
        try:
            f_true,f_recon = load_images_true(index=i)
            crop_true,crop_recon = random_crop(f_true,f_recon)
            toss_into_folder(crop_recon,crop_true,trainf,str(train_count))
            MSE_train.append(MSE(crop_true,crop_recon))
            train_count+=1
            if (i) % print_len ==0:
                toc = time.time()
                print('Finished ' + str(i) + ' out of ' + str(ntrain) +'. in ' + str(toc-tic) + ' s')
                tic = time.time()
        except Exception:print('Failed to load image ' + str(i))

    MSE_test = []

    for i in range(ntest):
        try:
            f_true,f_recon = load_images_true(index=i+ntrain)
            crop_true,crop_recon = random_crop(f_true,f_recon)
            toss_into_folder(crop_recon,crop_true,testf,str(test_count))
            MSE_test.append(MSE(crop_true,crop_recon))
            test_count+=1
            del f_true,f_recon,crop_true,crop_recon
            if (i) % print_len ==0:
                print('Finished ' + str(i+ntrain) + ' out of ' + str(ntest))
        except Exception:print('Failed to load image ' + str(i))

    for i in range(ntestFinal):
        try:
            f_true,f_recon = load_images_true(index=i+ntrain+ntest)
            crop_true,crop_recon = random_crop(f_true,f_recon)
            toss_into_folder(crop_recon,crop_true,testFinalf,str(test_count))
#             MSE_test2.append(MSE(crop_true,crop_recon))
            test_count+=1
            del f_true,f_recon,crop_true,crop_recon
            if (i) % print_len ==0:
                print('Finished ' + str(i+ntrain+ntest) + ' out of ' + str(ntest))
        except Exception:print('Failed to load image ' + str(i))

    MSE_train = np.asarray(MSE_train)
    MSE_test = np.asarray(MSE_test)

    print('Avg mse train: ' + str(np.mean(MSE_train)))
    print('Avg mse test: ' + str(np.mean(MSE_test)))
    print('std train: ' + str(np.std(MSE_train)))
    train_std = np.std(MSE_train)
    train_mean = np.mean(MSE_train)

    if AD_path is not None:

        MSE_train = []
        MSE_test = []
        mypath = AD_path
        from os import listdir
        from os.path import isfile, join
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

        # number_AD_files = np.min((len(onlyfiles),1000))
        number_AD_files = len(onlyfiles)

        skipped = 0
        total = 0

        save_mses = []


#         num_files = int(number_AD_files*percentage_train)
        for i in range(int(ntrain/10)):
            skip = False
            try:
                with open(AD_path + str(i) + '.pkl', 'rb') as f:
                    data = pickle.load(f)
                    X_train = data['X_train']
                    Y_train = data['y_train']
            except Exception:
                print('Failed to load : ' + AD_path + str(i))
                skip = True
            if not skip:
                want_new_one = False
                count_new_one = 0
                for j in range(np.asarray(X_train).shape[0]):
                    err = MSE(X_train[j],Y_train[j])
                    save_mses.append(err)
                    if err < train_mean + train_std*num_std_away:
                        toss_into_folder(X_train[j],Y_train[j],trainf,str(train_count))
                        MSE_train.append(MSE(X_train[j],Y_train[j]))
                        train_count+=1
                    total+=1

            if (i) % print_len ==0:
                toc = time.time()
                print('Finished ' + str(i) + ' out of ' + str(number_AD_files) +'. in ' + str(toc-tic) + ' s')
                tic = time.time()

        for i in range(int(ntest/10)):
            skip = False
            try:
                with open(AD_path + str(int(ntrain/10)+i)+'.pkl', 'rb') as f:
                    data = pickle.load(f)
                    X_train = data['X_train']
                    Y_train = data['y_train']
            except Exception:
                print('Failed to load : ' + AD_path + str(int(ntrain/10)+i))
                skip = True
            if not skip:
                for j in range(np.asarray(X_train).shape[0]):
                    err = MSE(X_train[j],Y_train[j])
                    save_mses.append(err)
                    if err < train_mean + train_std*num_std_away:
                        toss_into_folder(X_train[j],Y_train[j],testf,str(test_count))
                        MSE_test.append(MSE(X_train[j],Y_train[j]))
                        test_count+=1
                    total+=1
            if (i) % print_len ==0:
                toc = time.time()
                print('Finished ' + str(i) + ' out of ' + str(number_AD_files) +'. in ' + str(toc-tic) + ' s')
                tic = time.time()

        for i in range(int(ntestFinal/10)):
            skip = False
            try:
                with open(AD_path + str(int((ntrain+ntest)/10)+i)+'.pkl', 'rb') as f:
                    data = pickle.load(f)
                    X_train = data['X_train']
                    Y_train = data['y_train']
            except Exception:
                print('Failed to load : ' + AD_path + str(int((ntrain+ntest)/10)+i))
                skip = True
            if not skip:
                for j in range(np.asarray(X_train).shape[0]):
                    err = MSE(X_train[j],Y_train[j])
                    save_mses.append(err)
                    if err < train_mean + train_std*num_std_away:
                        toss_into_folder(X_train[j],Y_train[j],testFinalf,str(test_count))
#                         MSE_test.append(MSE(X_train[j],Y_train[j]))
                        test_count+=1
                    total+=1
            if (i) % print_len ==0:
                toc = time.time()
                print('Finished ' + str(i) + ' out of ' + str(number_AD_files) +'. in ' + str(toc-tic) + ' s')
                tic = time.time()

        MSE_train = np.asarray(MSE_train)
        MSE_test = np.asarray(MSE_test)

        print('AD Avg mse train: ' + str(np.mean(MSE_train)))
        print('AD Avg mse test: ' + str(np.mean(MSE_test)))
        print('AD std train: ' + str(np.std(MSE_train)))
        print('AD std test: ' + str(np.std(MSE_test)))

        print('Train count: ' + str(train_count) + '. Test count: ' + str(test_count) + '. Total: ' + str(total+ntrain+ntest))
        return save_mses


def generator_load_all(input_shape=(None,176,176,1),output_shape=(None,154,154,1),dataset='19',AD=False):
    main_directory = 'datasets/'
    if AD:
        ending='_AD/'
    else:
        ending='/'
    print('Generate training data!')
    # Train
    mypath = main_directory+ 'v' + dataset + '_'+ 'train' + ending
    print(mypath)
    X_train,Y_train = load_entire_dataset(mypath,input_shape,output_shape)
    # Test
    mypath = main_directory+ 'v' + dataset + '_'+ 'test' + ending
    print(mypath)
    X_test,Y_test = load_entire_dataset(mypath,input_shape,output_shape)
    
    return X_train,Y_train,X_test,Y_test

def load_entire_dataset(mypath,input_shape,output_shape):

    from os import listdir
    from os.path import isfile, join
    list_files = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]
    X = []
    Y = []
    print('file number:'+str(len(list_files)))
    for i in range(len(list_files)):
        with open(list_files[i], 'rb') as f:
            data = pickle.load(f)
            if len(data['X'].shape) ==2:
                data['X'] = data['X'].reshape(data['X'].shape[0],data['X'].shape[1],1)
            if len(data['Y'].shape) ==2:
                data['Y'] = data['Y'].reshape(data['Y'].shape[0],data['Y'].shape[1],1)
            
            X.append(fix_Ys(data['X'],input_shape))
            Y.append(fix_Ys(data['Y'],output_shape))
        
        # Fix up the X/Ys if they are not exactly the correct size
    print('---data load--')
    X = np.asarray(X)
    Y = np.asarray(Y)
    return X,Y


def generator(batch_size=64,input_shape=(None,176,176,1),output_shape=(None,154,154,1),dataset='19',AD=False,train=True):
    main_directory = 'datasets/'
    if AD:
        ending='_AD/'
    else:
        ending='/'
    
    if train:
        middle = 'train'
    else:
        middle = 'test'
    
    mypath = main_directory+ 'v' + dataset + '_'+ middle + ending
    return myGenerator(mypath,batch_size,input_shape,output_shape)

def fix_Ys(Y,cropped_output):
#     print(Y.shape)
#     print(cropped_output)
    y_cropped = np.zeros(cropped_output[1:])
    dif = int((Y.shape[1]-cropped_output[1])/2)
    y_cropped[:,:,:] = Y[dif:Y.shape[1]-dif,dif:Y.shape[1]-dif,:]
    return y_cropped

def myGenerator(mypath,batch_size,input_shape,output_shape):
    from os import listdir
    from os.path import isfile, join
    list_files = [mypath + f for f in listdir(mypath) if isfile(join(mypath, f))]

#     print(input_shape)
#     print(output_shape)
    # Infinite loop
    while 1:
        # Shuffle the list of files
        from random import shuffle
        shuffle(list_files)
        # Fix list of files to be divisible by batch_size
        i=0
        while len(list_files) % batch_size != 0:
            list_files.append(list_files[i])
            i+=1
        # For each slice of the shuffled files, where len(slice) == batch_size
        # Open files, read to a single array with first shape[0] == batch_size; yield data
        for i in range(int(len(list_files)/batch_size)):
            X = []
            Y = []
            for j in range(batch_size):
                with open(list_files[i*batch_size+j], 'rb') as f:
                    data = pickle.load(f)
                    if len(data['X'].shape) ==2:
                        data['X'] = data['X'].reshape(data['X'].shape[0],data['X'].shape[1],1)
                    if len(data['Y'].shape) ==2:
                        data['Y'] = data['Y'].reshape(data['Y'].shape[0],data['Y'].shape[1],1)
                    
                    X.append(fix_Ys(data['X'],input_shape))
                    Y.append(fix_Ys(data['Y'],output_shape))
            X = np.asarray(X)
            Y = np.asarray(Y)
            # Fix up the X/Ys if they are not exactly the correct size
            
            yield X,Y

 
#     model.add(MaxPooling2D(pool_size=pool_size))
    # model.add(Dropout(0.25))

#     model.add(Flatten())
#     model.add(Dense(128))
#     model.add(Activation('relu'))
    # model.add(Dropout(0.5))
#     model.add(Dense(nb_classes,name='final_dense'))
#     model.add(Activation('softmax',name ='softmax'))

#     model.load_weights("model_dropout.h5")



def perform_tv_projection(img,niter=20,tau=600):

    # takes gradient of 2d image with 1 color channel
    def gradient(img):
        sz = np.shape(img)
        if sz[0]!=sz[1]:
            print('X and Y dimension are not equal, gradient function will break.')
            sys.exit()

        fx = np.zeros(sz)
        fy = np.zeros(sz)
        for i in range(sz[0]-1):
            fx[i,:] = img[i+1,:] - img[i,:]
            fy[:,i] = img[:,i+1] - img[:,i]

        fx[sz[0]-1,:] = 0
        fy[:,sz[1]-1] = 0

        return fx,fy

    # perform vector field normalization - l2 norm, assumes 2d images with 1 color channels
    def perform_vf_normalization(vfx,vfy):
        n = np.sqrt(np.square(vfx)+np.square(vfy))
        n[n<np.power(10.,-9)] = 1
        return vfx/n,vfy/n

    # divergence (backward difference)
    def div(Px,Py):
        fx = np.zeros(np.shape(Px))
        fy = np.zeros(np.shape(Py))

        for i in range(np.shape(Px)[0]-1):
            fx[i+1,:] = Px[i,:] - Px[i+1,:]
            fy[:,i+1] = Py[:,i] - Py[:,i+1]

        fd = fx + fy
        return fd

    # We are assuming l2 norm - from the options
    def compute_total_variation(img):

        fx,fy = gradient(img)
        TV = np.sqrt( np.square(fx) + np.square(fy) )
        TV = np.sum(np.sum(TV))

        return TV

    x = img
    x0 = img
    # niter =50
    # tau = 5
    tau1 = compute_total_variation(x)
    err_tv = []

    if tau1 < tau:
        print('Done - exit function immediately because TV is already within constraint!')

    for i in range(niter):

        # subgradient of total variation
        tx,ty = gradient(x)
        # Normalize with respect to l2-norm
        tx,ty = perform_vf_normalization(tx,ty)

        # Take divergence
        t = div(tx,ty)

        # gradient projection onto TV=tau
        tau1 = compute_total_variation(x)

        reshaped_t = np.reshape(t,(np.cumprod(np.shape(t))[-1],))
        d = np.dot(reshaped_t,reshaped_t)

        if d > np.power(10.,-9):
            z = x - (tau1-tau)*t/d;
        else:
            z = x

        # Step 5, which minimizes over the intersection of two half spaces.
        new_shape = np.cumprod(np.shape(x))[-1]
        pi = np.dot(np.reshape(x0-x,new_shape),np.reshape(x-z,new_shape))
        mu = np.dot(np.reshape(x0-x,new_shape),np.reshape(x0-x,new_shape))
        nu = np.dot(np.reshape(x-z,new_shape),np.reshape(x-z,new_shape))
        rho = mu*nu-np.square(pi)

        if rho==0 and pi>=0:
            x=z
        elif rho>0 and pi*nu >=rho:
            x = x0 + (1+pi/nu)*(z-x)
        elif rho>0 and pi*nu<rho:
            x = x + nu/rho*(pi*(x0-x) + mu*(z-x))
        else:
            print('Error PBM')

        # record errors
        err_tv.append(tau1-tau)
        print('Error at iteration: ' +str(i) + ' : ' + str(tau1-tau))

        if tau1<tau:
            break

    return x

# EOF #
