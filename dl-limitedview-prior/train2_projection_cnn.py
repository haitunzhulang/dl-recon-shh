########## Train Projection CNN, generating outputs every epoch or so

# This is the same as train_projection_cnn.py except training with entire image, 
# and using data augmentation to shrink the size of the images going into the CNN

# nohup python3 train2_projection_cnn.py --nb_filters=128 --f_dim1 16 --f_dim2 3 --f_dim3 10 --longer 2 --nb_epochs 500 --loss mse --batch_size 32 --dropout .3 2>&1&
# nohup python3 train2_projection_cnn.py --nb_filters=128 --f_dim1 9 --f_dim2 3 --f_dim3 7 --longer 1 --nb_epochs 500 --loss mse --batch_size 32  2>&1&
# nohup python3 train2_projection_cnn.py --nb_filters=128 --f_dim1 9 --f_dim2 3 --f_dim3 7 --longer 1 --nb_epochs 500 --loss mse --batch_size 32 --dropout .3 2>&1&


# nohup python3 train2_projection_cnn.py --nb_filters=128 --f_dim1 9 --f_dim2 3 --f_dim3 7 --longer 1 --nb_epochs 500 --loss mse --batch_size 32 --dropout .3 --normalize True --final_act True 2>&1&

# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --nb_epochs 1000 --loss mse --batch_size 32 --final_act True --relu True --normalize True --dataset 9 2>&1& --- RELU IS WORSE THAN TANH
# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --nb_epochs 500 --loss mse --batch_size 32 --final_act True --normalize True --dataset 9 2>&1&

# v10 -- Randome Initialization
# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --nb_epochs 1000 --loss mse --batch_size 32 --final_act True --normalize True --dataset 10 --longer 2 2>&1&
# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --nb_epochs 1000 --loss mse --batch_size 32 --final_act True --normalize True --dataset 10  2>&1&

# v11
# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --lr .0005 --loss mse --batch_size 20 --final_act True --dataset 11 --nb_epochs 500 --longer 2 2>&1&


# v10
# nohup python3 train2_projection_cnn.py --nb_filters=512 --f_dim1 12 --f_dim2 3 --f_dim3 8 --lr .0005 --loss mse --batch_size 12 --final_act True --dataset 11 --nb_epochs 500 --longer 4 2>&1&



from keras.callbacks import Callback
import helper_functions as hf
import CNN_generator as cg
import numpy as np
np.random.seed(1337)
import argparse
import random
from keras.optimizers import Adam
from keras import backend as K
import os
import contextlib



output_folder = 'projection_results4/'
class generateImageCallback(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.lrs = []

    def on_epoch_end(self, epoch,logs={}):
        from PIL import Image
        self.losses.append(logs.get('loss'))
        self.lrs.append(K.get_value(model.optimizer.lr))
        # save_predictions
        image = self.generate_image()
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(output_folder+  \
            self.name + '/singles_' + str(epoch) +".png")
        
        epoch += self.num_epochs
        save_len = 50
        print(logs)
        if epoch<5:
            self.save_best()
            self.best_loss = logs['loss']
        elif logs['loss'] < self.best_loss: # improvement, save it!
            self.save_best()
            self.best_loss=logs['loss']
        elif logs['loss']>1.1*self.best_loss: ## loss has exploded!
            print('Reloading Weight file, as loss exploded')
            self.load_best()
            self.lower_learning_rate()
            
        if epoch%save_len==0:
            self.save_best(name=str(epoch))
    
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
            decay= .75
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
    parser.add_argument("--normalize_simplistic",type=bool,default=False)
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


for i in range(len(f_dim1)):

    ## Load model
    if args.dataset==6:
        input_shape=(64,64,1)
    elif args.dataset < 11:
        input_shape=(196,196,1)
    else:
        input_shape=(176,176,1)
    model = cg.projection_network(input_shape=input_shape,relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer,dropout=args.dropout)
    model.name = 'deartifact_'+str(lrelu[i])+'_'+'{0:.3f}'.format(alpha[i])+ '_' + str(f_dim1[i]) + \
            '_' + str(f_dim2[i]) + '_' + str(f_dim3[i])+ '_' + str(nb_filters[i])+ \
            '_' + str(elu[i]) + '_' + str(lr[i])+ '_' + args.loss+ '_' + str(args.longer)+ \
            '_' + str(args.nb_epochs) + '_' + str(args.normalize)+ '_'+ str(args.dropout) + \
            '_' + str(args.final_act)+'_' + str(args.relu) +'_' + str(args.dataset)+'_regular'
    print('model name: ' + model.name)
    adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(loss=args.loss, optimizer=adam)

    hf.generate_folder(output_folder+model.name)
    ## Load dataset
    X_train,X_test,Y_train,Y_test = hf.load_data(version=args.dataset,normalize_projection=args.normalize,\
                normalize=False,normalize_simplistic = args.normalize_simplistic)
                
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
    cb.set_inputs(X_test[0:8,:,:,:])
    cb.set_name(model.name)

    ##  Run model
    # X_train = X_train[1:200,:,:,:]
    # Y_train = Y_train[1:200,:,:,:]
    cb.set_num_epochs(0)
    val_loss =  hf.run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=args.nb_epochs,batch_size=args.batch_size,DA=False,callback=cb,save_every=False,verbose=args.verbose)
    
    lrs = []
    for lr in cb.lrs: lrs.append(lr)
    fn = output_folder + model.name + '/lr_log.txt'
    try:
        os.remove(fn)
    except Exception:
        pass
    
    thefile = open(fn, 'w')
    for lr in lrs:
        thefile.write("%s\n" % str(lr))

    thefile.close()


##
















# EoF #
