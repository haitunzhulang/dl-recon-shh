########## Train Projection CNN, generating outputs every epoch or so

# This is the same as train3_projection_cnn.py except for the looping, augment dataset DA, 
# idea, we are doing a projection with the CNN, and the LS optimization for a while (250 iterations?)


# nohup python3 train4_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 40 --final_act True --dataset 13 --init_epochs 500 --inter_epochs 50 --num_loop 5 --lr .0001 --lower_learning_rate True --DA_Lr_decrease .1 2>&1&

# Mar 9 -- randomizing the AD idea
# nohup python3 train4_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 40 --final_act True --dataset 13 --init_epochs 1 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True 2>&1&

# Mar 13 -- correctly not lowering lr during inter phase, also increasing num_add
# nohup python3 train4_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 40 --final_act True --dataset 13 --init_epochs 1 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 2>&1&








from keras.callbacks import Callback
import helper_functions as hf
import CNN_generator as cg
import numpy as np
np.random.seed(1337)
import argparse
import random
from keras.optimizers import SGD,Adam
import contextlib
import os
import contextlib
import math

from keras import backend as K



output_folder = 'projection_results5/'
hf.generate_folder(output_folder)
hf.generate_folder('models/')
class generateImageCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_losses = []
        self.lrs = []
        

    def on_epoch_end(self, epoch,logs={}):
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.lrs.append(K.get_value(model.optimizer.lr))
        # save_predictions
#         image = self.generate_image()
#         image = image*127.5+127.5
        epoch += self.num_epochs
#         Image.fromarray(image.astype(np.uint8)).save(output_folder+  \
#             self.name + '/singles_' + str(epoch) +".png")
        
        save_iter=45
        save_len = 10
        print(logs)
        if epoch<10:
            self.save_best()
            self.best_loss = logs['loss']
        elif logs['loss'] < self.best_loss and epoch%save_iter==0: # improvement, save it!
            self.save_best()
            self.best_loss=logs['loss']
        elif logs['loss']>2*self.best_loss and epoch < 40: ## loss has exploded!
            print('Reloading Weight file, as loss exploded')
            self.load_best()
            self.lower_learning_rate()
            
        if epoch%save_len==0:
            self.save_best(name=str(epoch))
            
        # Save training plot
        hf.save_training_plot(self.losses,self.val_losses,self.lrs,model_name=self.model.name,plot_folder=output_folder,suffix=self.training_nums_suffix)
        np.savetxt(output_folder + self.model.name + '/training_nums' +self.training_nums_suffix + '.out', (self.losses,self.val_losses,self.lrs), delimiter=',')
        
    def set_training_nums_suffix(self,x):
        self.training_nums_suffix=str(x)
    
    def randomly_increase_learning_rate(self):
        random_chance = .02
        increase_rate = 2
        
        if random.random()<random_chance:
            decay= increase_rate
            K.set_value(self.model.optimizer.lr, decay * K.get_value(self.model.optimizer.lr))
            print('Increasing learning rate to : ' + str(K.get_value(self.model.optimizer.lr)))
    
    def lower_learning_rate(self):
        min_learning_rate = .00000001
        if K.get_value(self.model.optimizer.lr) > min_learning_rate:
            decay= 1
            K.set_value(self.model.optimizer.lr, decay * K.get_value(self.model.optimizer.lr))
            print('Lowering learning rate to : ' + str(K.get_value(self.model.optimizer.lr)))
        
    def load_best(self):
        weight_file = output_folder + self.name + '/best.h5'
        self.model.load_weights(weight_file)
            
    def save_best(self,name='best'):
        hf.save_model(self.model,self.name +'/' + name,folder=output_folder)
        
    def set_num_epochs(self,num_epochs):
        self.num_epochs=num_epochs

    def set_inputs(self,inputs):
        self.inputs=inputs

    def set_name(self,name):
        self.name=name

    def generate_image(self):
        generated_images = self.model.predict(self.inputs, verbose=0)
        image = self.combine_images(generated_images)
        return image

    def combine_images(self,generated_images):
        import math
        num = generated_images.shape[0]
        width = int(math.sqrt(num))
        height = int(math.ceil(float(num)/width))
        shape = generated_images.shape[1:3]
        image = np.zeros((height*shape[0], width*shape[1]),
                         dtype=generated_images.dtype)
        for index, img in enumerate(generated_images):
            i = int(index/width)
            j = index % width
            image[i*shape[0]:(i+1)*shape[0], j*shape[1]:(j+1)*shape[1]] = \
                img[:, :,0]
        return image


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float,default=.3)
    parser.add_argument("--lrelu", type=bool, default=False)
    parser.add_argument("--elu",type=bool,default=False)
    parser.add_argument("--f_dim1", type=int, default=16)
    parser.add_argument("--f_dim2", type=int, default=1)
    parser.add_argument("--f_dim3", type=int, default=8)
    parser.add_argument("--nb_filters", type=int, default=512)
    parser.add_argument("--nb_epochs", type=int, default=100)
    parser.add_argument("--nb_searches",type=int,default=25)
    parser.add_argument("--explore",type=bool,default=False)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--loss",type=str,default='mse')
    parser.add_argument("--verbose",type=int,default=0)
    parser.add_argument("--longer",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--normalize",type=bool,default=False)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--final_act",type=bool,default=False)
    parser.add_argument("--relu",type=bool,default=False)
    parser.add_argument("--dataset",type=int,default=6)
    parser.add_argument("--init_epochs",type=int,default=50)
    parser.add_argument("--inter_epochs",type=int,default=5)
    parser.add_argument("--num_loop",type=int,default=10)
    parser.add_argument("--num_add",type=int,default=150)
    parser.add_argument("--normalize_simplistic",type=bool,default=False)
    parser.add_argument("--lower_learning_rate",type=bool,default=False)
    parser.add_argument("--lr_step",type=int,default=500)
    parser.add_argument("--lr_drop_pc",type=float,default=0.5)
    parser.add_argument("--DA_Lr_decrease",type=float,default=0.1)
    parser.add_argument("--use_previous_best",type=bool,default=False)
    
    args = parser.parse_args()
    return args

args = get_args()
print(args)

alpha = []
lrelu = []
f_dim1 = []
f_dim2 = []
f_dim3 = []
nb_filters = []
elu = []
lr = []


if args.explore is False:
    alpha.append(args.alpha)
    lrelu.append(args.lrelu)
    f_dim1.append(args.f_dim1)
    f_dim2.append(args.f_dim2)
    f_dim3.append(args.f_dim3)
    nb_filters.append(args.nb_filters)
    elu.append(args.elu)
    lr.append(args.lr)
    oddity = (int(f_dim1[0]%2 ==1) + int(f_dim2[0]%2 ==1) + int(f_dim3[0]%2 ==1))
    if oddity == 2  or oddity==0:
            f_dim3[0]+=1
else:
    print('Explore may not work correctly - feb16')
    for i in range(args.nb_searches):
        f_dim1.append(random.randint(8,16))
        f_dim2.append(1)#f_dim2.append(random.randint(1,11))
        f_dim3.append(random.randint(8,16))
        oddity = (int(f_dim1[0]%2 ==1) + int(f_dim2[0]%2 ==1) + int(f_dim3[0]%2 ==1))
        if oddity ==2 or oddity ==0:
            f_dim3[i]+=1
        alpha.append(random.random()*(1/10)+.15)
        lrelu.append(args.lrelu)
        nb_filters.append(512)#nb_filters.append(random.randint(512-128,512+128))
        elu.append(args.elu)
        lr.append(args.lr)

# try:
for i in range(len(f_dim1)):

    ## Load model
    if args.dataset==6:
        input_shape=(64,64,1)
    elif args.dataset<11:
        input_shape=(196,196,1)
    else:
        input_shape=(176,176,1)
    
    model = cg.projection_network(input_shape=input_shape,relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer,dropout=args.dropout)
    model.name = 'deartifact_'+str(lrelu[i])+'_'+'{0:.3f}'.format(alpha[i])+ '_' + str(f_dim1[i]) + \
            '_' + str(f_dim2[i]) + '_' + str(f_dim3[i])+ '_' + str(nb_filters[i])+ \
            '_' + str(elu[i]) + '_' + str(lr[i])+ '_' + args.loss+ '_' + str(args.longer)+ \
            '_' + str(args.nb_epochs) + '_' + str(args.inter_epochs)+'_' + str(args.normalize)+ '_'+ str(args.dropout) + \
            '_' + str(args.final_act)+'_' + str(args.relu) +'_' + str(args.dataset)+ \
            '_' + str(args.normalize_simplistic) + '_' + str(args.num_loop) + \
            '_' + str(args.lower_learning_rate) + '_' + str(args.lr_step) + \
            '_' + str(args.lr_drop_pc) + '_' + str(args.DA_Lr_decrease) + '_' +str(args.num_add)+ '_march8'
    print('model name: ' + model.name)
    
    if args.use_previous_best:
        best = hf.load_trained_CNN(name=model.name + '/390',folder=output_folder)
        model.set_weights(best.get_weights())
        import time
        # rename files (which will be overwritten) to old/todays 
        fold = output_folder + model.name + '/'
        os.system('mv ' + fold + 'training_nums.out ' + fold + 'training_nums_' + time.strftime("%d_%m_%Y_%H_%M_%S") +'.out')
        os.system('mv ' + fold + 'training_plot.png ' + fold + 'training_plot_' + time.strftime("%d_%m_%Y_%H_%M_%S") +'.png')
    
    sgd = Adam(lr=args.lr)
    model.compile(loss=args.loss, optimizer=sgd)
    ## Load dataset
    X_train,X_test,Y_train,Y_test = hf.load_data(version=args.dataset,normalize_projection=args.normalize,\
                normalize=False,normalize_simplistic=args.normalize_simplistic)
            

    hf.generate_folder(output_folder+model.name)
    # X_train = X_train[1:100,:,:,:]
#     Y_train = Y_train[1:100,:,:,:]

    ## Fix Y's of dataset to be cropped to middle square from output of model
    cropped_output=model.output_shape
    print ('output shape : ' + str(cropped_output))

    def fix_Ys(Y,cropped_output=cropped_output):
        y_cropped = np.zeros((Y.shape[0],)+cropped_output[1:])
        dif = int((Y.shape[2]-cropped_output[2])/2)
        for j in range(Y.shape[0]):
            y_cropped[j,:,:,:] = Y[j,dif:Y.shape[1]-dif,dif:Y.shape[1]-dif,:]

        return y_cropped

    Y_train = fix_Ys(Y_train)
    Y_test = fix_Ys(Y_test)



    cb = generateImageCallback()
    cb.set_training_nums_suffix('')
    cb.set_inputs(X_test[0:8,:,:,:])
    cb.set_name(model.name)

    ##  Run model
    #X_train = X_train[1:200,:,:,:]
    #Y_train = Y_train[1:200,:,:,:]

    # do for 50 epochs
    init_epochs = args.init_epochs
    cb.set_num_epochs(0)
    val_loss =  hf.run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=init_epochs,batch_size=args.batch_size,DA=False,callback=cb,save_every=False,verbose=args.verbose,lower_learning_rate=args.lower_learning_rate,lr_step=args.lr_step,lr_drop_pc=args.lr_drop_pc)
    cb.set_num_epochs(init_epochs)
    lrs = []
    losses = []
    val_losses = []
    for loss in cb.losses: losses.append(loss)
    for val_loss in cb.val_losses: val_losses.append(val_loss)
    for lr in cb.lrs: lrs.append(lr)
    # add to training set, do for 3 more epochs, then repeat
    orig_x_train_shape = X_train.shape
    for j in range(args.num_loop):
        inter_epochs = args.inter_epochs
        cb.set_num_epochs(init_epochs+j*inter_epochs)
        num_add = args.num_add
        #print(model.output_shape)
        size_dif = np.asarray(model.input_shape[1:4])-np.asarray(model.output_shape[1:4])
        #print(size_dif)

        model_larger = cg.projection_network(input_shape=(256,256,1),relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer,dropout=args.dropout)
        model_larger.set_weights(model.get_weights())
        model_larger.name = model.name

        X_train = list(X_train)
        Y_train = list(Y_train)
        lr = K.get_value(model.optimizer.lr)
        cropped_output=model.output_shape
        for k in range(num_add):
            index = random.randint(0,orig_x_train_shape[0])
            if args.dataset == 13:
                theta=60
                data_dirname='../xct-parallelbeam-matlab/dataset_v13_60_noRI_scale_nonneg/'
            else:
                raise ValueError('Used dataset other than v13, must specify theta of this dataset for looping')
            
            H,g,f_true,f_recon = hf.load_H_g_target(version=2,index=index,theta=theta,dirname=data_dirname)
            cutoff=np.random.random()*.02 #max cutoff is .01
            niters= math.floor(np.random.random()*1000)
            
            ls_starting_with_projection,psnr_save,projection_times_save,magnitudes_save = hf.Projected_PLS(model_larger,cutoff=cutoff,niters=niters,version=2,theta=theta,index=index,display=False,data_dirname=data_dirname)
            Y_inter = []
            X_inter = []
            X_inter.append(ls_starting_with_projection.reshape(256,256,1))
            Y_inter.append(f_true.reshape(256,256,1))
            X_inter = np.asarray(X_inter)
            Y_inter = np.asarray(Y_inter)
            #print(Y_inter.shape)
            #print(X_inter.shape)
            X_inter = fix_Ys(X_inter,model.input_shape)
            Y_inter = fix_Ys(Y_inter,model.output_shape)
            
            X_train.append(X_inter[0])
            Y_train.append(Y_inter[0]) # hopefully this is the same as Y_train[index]...
            

        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
        fname = 'datasets/' + model_larger.name + '_' + str(j) + '.pkl'
        import pickle
        data = {}
        data['X_train'],data['X_test'],data['y_train'],data['y_test'] = X_train,X_test,Y_train,Y_test
        with open(fname, 'wb') as f:
            pickle.dump(data,f)
        
        model = cg.projection_network(input_shape=input_shape,relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer,dropout=args.dropout)
        model.set_weights(model_larger.get_weights())
        model.name = model_larger.name
        adam = Adam(lr=args.DA_Lr_decrease*lrs[-1])
        model.compile(loss=args.loss, optimizer=sgd)
        cb.set_training_nums_suffix(j)
        cb.set_num_epochs(init_epochs+j*inter_epochs)
        
        hf.run_model(model,np.asarray(X_train),np.asarray(Y_train),X_test,Y_test,nb_epoch=inter_epochs,batch_size=args.batch_size,DA=False,callback=cb,save_every=False,verbose=args.verbose,lower_learning_rate=args.lower_learning_rate,lr_step=args.lr_step,lr_drop_pc=args.lr_drop_pc)
        for lr in cb.lrs: lrs.append(lr)
        for loss in cb.losses: losses.append(loss)
        for val_loss in cb.val_losses: val_losses.append(val_loss)
    
        fn = output_folder + model.name + '/lr_log.txt'
        try:
            os.remove(fn)
        except Exception:
            pass
    
        thefile = open(fn, 'w')
        for lr in lrs:
            thefile.write("%s\n" % str(lr))

        thefile.close()
    

    hf.save_model(model,model.name)
    hf.save_training_plot(losses,val_losses,lrs,model_name=model.name,plot_folder=output_folder)
# except Exception as e:
#     import yagmail
#     yag = yagmail.SMTP('brave.rebel7@gmail.com', 'testchpc1')
#     body = 'Training CNN failed! \n \n  Args: ' + str(args) + ' \n Exception: ' + str(e)
#     yag.send('bmkelly@mail.bradley.edu', 'CNN Training failed for some reason', body)
    
    



##
















# EoF #
