########## Train Projection CNN, generating outputs every epoch or so

# This is the same as train2_projection_cnn.py except training with the looping argument:
# after X epochs, we start adding Y training images every Z epochs.

# v1
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 32 --final_act True --dataset 9 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 9 --f_dim2 3 --f_dim3 5 --loss mse --batch_size 32 --final_act True  --dataset 10 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 5 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 9 2>&1&

# v11
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 32 --final_act True --dataset 11 2>&1&
# nohup python3 train2_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 20 --final_act True --dataset 11 --nb_epochs 500 --longer 2 2>&1&

# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 64 --final_act True --dataset 11 --verbose 1 --init_epochs 75 --normalize_simplistic True 2>&1&

# v10 - feb 23
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 64 --final_act True --dataset 11 --verbose 1 --init_epochs 50 --inter_epochs 5 2>&1&

# v12 - feb 24
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 64 --final_act True --dataset 12 --init_epochs 50 --inter_epochs 5 --num_loop 12 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 12 --f_dim2 3 --f_dim3 8 --longer 2 --loss mae --batch_size 32 --final_act True --dataset 12 --init_epochs 50 --inter_epochs 5 --lr .01 2>&1&

# v13
# nohup python3 train3_projection_cnn.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 128 --final_act True --dataset 13 --init_epochs 50 --inter_epochs 10 --num_loop 12 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=256 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 64 --final_act True --dataset 13 --init_epochs 50 --inter_epochs 10 --num_loop 12 --lr .01 2>&1&

# deeper v13 -feb 28
# nohup python3 train3_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 13 --init_epochs 50 --inter_epochs 10 --num_loop 12 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 13 --init_epochs 50 --inter_epochs 10 --num_loop 12 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=512 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 16 --final_act True --dataset 13 --init_epochs 50 --inter_epochs 10 --num_loop 12 --lr .01 2>&1&


# deeper v13 -- waiting longer for lr drops. -mar 1
# nohup python3 train3_projection_cnn.py --nb_filters=512 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 13 --init_epochs 100 --inter_epochs 50 --num_loop 8 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 13 --init_epochs 100 --inter_epochs 50 --num_loop 8 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 32 --final_act True --dataset 13 --init_epochs 500 --inter_epochs 50 --num_loop 0 --lr .01 2>&1&


# v13 trained longer, but shallower
# nohup python3 train3_projection_cnn.py --nb_filters=128 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 64 --final_act True --dataset 13 --init_epochs 5000 --inter_epochs 50 --num_loop 0 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=128 --f_dim1 16 --f_dim2 1 --f_dim3 8 --longer 2 --loss mse --batch_size 64 --final_act True --dataset 13 --init_epochs 3000 --inter_epochs 200 --num_loop 10 --lr .01 2>&1&

# v13 trained longer, but shallower even shallower
# nohup python3 train3_projection_cnn.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 128 --final_act True --dataset 13 --init_epochs 5000 --inter_epochs 50 --num_loop 0 --lr .01 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 128 --final_act True --dataset 13 --init_epochs 3000 --inter_epochs 200 --num_loop 10 --lr .01 2>&1&



# mar7 - train with Adam, train w/ lr dropping and without for 1k iterations.
# nohup python3 train3_projection_cnn.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 40 --final_act True --dataset 13 --init_epochs 1000 --inter_epochs 50 --num_loop 0 --lr .001 --lower_learning_rate True 2>&1&
# nohup python3 train3_projection_cnn.py --nb_filters=512 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 40 --final_act True --dataset 13 --init_epochs 1000 --inter_epochs 50 --num_loop 0 --lr .001 --lower_learning_rate True 2>&1&



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

from keras import backend as K



output_folder = 'projection_results4/'
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
        hf.save_training_plot(self.losses,self.val_losses,self.lrs,model_name=self.model.name,plot_folder=output_folder)
        np.savetxt(output_folder + self.model.name + '/training_nums.out', (self.losses,self.val_losses,self.lrs), delimiter=',')
        
    
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
    parser.add_argument("--num_add",type=int,default=500)
    parser.add_argument("--normalize_simplistic",type=bool,default=False)
    parser.add_argument("--lower_learning_rate",type=bool,default=False)
    parser.add_argument("--lr_step",type=int,default=500)
    parser.add_argument("--lr_drop_pc",type=float,default=0.5)
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
            '_' + str(args.nb_epochs) + '_' + str(args.normalize)+ '_'+ str(args.dropout) + \
            '_' + str(args.final_act)+'_' + str(args.relu) +'_' + str(args.dataset)+ \
            '_' + str(args.normalize_simplistic) + '_' + str(args.num_loop) + \
            '_' + str(args.lower_learning_rate) + '_ ' + str(args.lr_step) + \
            '_' + str(args.lr_drop_pc) + '_' + '_march7'
    print('model name: ' + model.name)
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
    for j in range(args.num_loop):
        inter_epochs = args.inter_epochs
        cb.set_num_epochs(init_epochs+j*inter_epochs)
        num_add = args.num_add
        #print(model.output_shape)
        size_dif = np.asarray(model.input_shape[1:4])-np.asarray(model.output_shape[1:4])
        #print(size_dif)

        model_larger = cg.projection_network(input_shape=tuple(np.asarray(input_shape)+size_dif),relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer,dropout=args.dropout)
        model_larger.set_weights(model.get_weights())

        x_train_shape = X_train.shape
        X_train = list(X_train)
        Y_train = list(Y_train)
        lr = K.get_value(model.optimizer.lr)

        for k in range(num_add):
            index = random.randint(0,x_train_shape[0])
            padded = np.zeros(tuple(np.asarray((1,) + x_train_shape[1:4])+[0,size_dif[1],size_dif[1],0]))
            dif=int(size_dif[1]/2)
            padded[:,dif:x_train_shape[1]+dif,dif:x_train_shape[1]+dif,:] = X_train[index][:,:,:]
            pred = model_larger.predict(padded)
            X_train.append(pred.reshape(model.input_shape[1],model.input_shape[2],1))
            Y_train.append(Y_train[index])

        X_train = np.asarray(X_train)
        Y_train = np.asarray(Y_train)
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
