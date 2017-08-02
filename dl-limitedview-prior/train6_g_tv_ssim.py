########## Train Projection CNN, generating outputs every epoch or so

# This is the same as train3_projection_cnn.py except we are stacking the CNN once and linking the
# weights.


# Mar 14
# nohup python3 train6_g_tv_ssim.py 2>&1&
# nohup python3 train6_g_tv_ssim.py --lr .01 --nb_filters 128 --batch_size 32 --verbose 1 2>&1&


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



output_folder = 'projection_results6/'
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
    parser.add_argument("--nb_filters", type=int, default=32)
    parser.add_argument("--nb_epochs", type=int, default=100)
    parser.add_argument("--nb_searches",type=int,default=25)
    parser.add_argument("--explore",type=bool,default=False)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--loss",type=str,default='mse')
    parser.add_argument("--verbose",type=int,default=0)
    parser.add_argument("--longer",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=128)
    parser.add_argument("--normalize",type=bool,default=False)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--final_act",type=bool,default=False)
    parser.add_argument("--relu",type=bool,default=False)
    parser.add_argument("--dataset",type=int,default=16)
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
    parser.add_argument("--scale_aux_loss",type=float,default=1.0)
    parser.add_argument("--num_stacks",type=int,default=1)
    parser.add_argument("--equal_AD",type=bool,default=False)
    parser.add_argument("--num_normal_training_examples",type=int,default=1000)
    
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
    input_shape=(176,176,1)
    if args.dataset==6:
        input_shape=(64,64,1)
    elif args.dataset<11:#
        input_shape=(196,196,1)
    elif args.dataset==16:
        input_shape=(100,256,1)
    model = cg.g_tv_to_ssim_network(nb_filters=args.nb_filters)
    model.name = 'g_tv_ssim' + '_' + str(lr[i])+ '_' + args.loss+ '_' + \
            '_' + str(args.nb_epochs) + '_nbfilters' + str(args.nb_filters) + '_' + str(args.dataset)+ \
            '_' + str(args.lower_learning_rate) + '_' + str(args.lr_step) + \
            '_' + str(args.lr_drop_pc) + '_' + str(args.DA_Lr_decrease) + '_march14'
    print('model name: ' + model.name)
    
    
    adam = Adam(lr=args.lr)
#     print(loss_weights)
    model.compile(loss=args.loss, optimizer=adam)
    ## Load dataset
    X_train,X_test,Y_train,Y_test,tv_train,tv_test = hf.load_data(version=args.dataset)

    hf.generate_folder(output_folder+model.name)

    ## Fix Y's of dataset to be cropped to middle square from output of model



    cb = generateImageCallback()
    cb.set_training_nums_suffix('')
    cb.set_name(model.name)

    ##  Run model
    #X_train = X_train[1:2,:,:,:]
    #Y_train = Y_train[1:2,:,:,:]

    init_epochs = args.init_epochs
    cb.set_num_epochs(0)
    val_loss =  hf.run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=args.nb_epochs,batch_size=args.batch_size,DA=False,callback=cb,save_every=False,verbose=args.verbose,lower_learning_rate=args.lower_learning_rate,lr_step=args.lr_step,lr_drop_pc=args.lr_drop_pc,tv_train=tv_train,tv_test=tv_test,g_tv_ssim=True)
    
    
    hf.save_model(model,model.name)
    hf.save_training_plot(cb.losses,cb.val_losses,cb.lrs,model_name=model.name,plot_folder=output_folder)
# except Exception as e:
#     import yagmail
#     yag = yagmail.SMTP('brave.rebel7@gmail.com', 'testchpc1')
#     body = 'Training CNN failed! \n \n  Args: ' + str(args) + ' \n Exception: ' + str(e)
#     yag.send('bmkelly@mail.bradley.edu', 'CNN Training failed for some reason', body)
    
    



##
















# EoF #
