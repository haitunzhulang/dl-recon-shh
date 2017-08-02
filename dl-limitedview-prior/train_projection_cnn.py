########## Train Projection CNN, generating outputs every epoch or so

# nohup python3 train_projection_cnn.py --explore True --elu True --loss mae  2>&1&
# nohup python3 train_projection_cnn.py --explore True --lrelu True 2>&1&
# nohup python3 train_projection_cnn.py --explore True --loss mae 2>&1&

# nohup python3 train_projection_cnn.py --f_dim1 16 --f_dim3 8 --elu True --alpha .35 --longer 3 --nb_epochs 1000 2>&1&
# nohup python3 train_projection_cnn.py --f_dim1 16 --f_dim3 8 --elu True --alpha .35200 --longer 2 --nb_epochs 200 2>&1&
# nohup python3 train_projection_cnn.py --f_dim1 16 --f_dim3 8 --elu True --alpha .349 --nb_epochs 3000 2>&1&

# nohup python3 train_projection_cnn.py --nb_filters 64 --f_dim1 9 --f_dim2 3 --f_dim3 5 --longer 2 --nb_epochs 500 --batch_size 512 2>&1&
# nohup python3 train_projection_cnn.py --nb_filters 64 --f_dim1 16 --f_dim2 3 --f_dim3 8 --longer 2 --nb_epochs 500 --loss mae --batch_size 256 2>&1&

# nohup python3 train_projection_cnn.py --nb_filters 64 --f_dim1 16 --f_dim2 3 --f_dim3 8 --longer 2 --nb_epochs 500 --loss mae --batch_size 256 --normalize False 2>&1&


from keras.callbacks import Callback
import helper_functions as hf
import CNN_generator as cg
import numpy as np
np.random.seed(1337)
import argparse
import random
from keras.optimizers import Adam
from keras import backend as K



output_folder = 'projection_results2/'
class generateImageCallback(Callback):

	
    def on_train_begin(self, logs={}):
        self.losses = []


    def on_epoch_end(self, epoch,logs={}):
        from PIL import Image
        self.losses.append(logs.get('loss'))
        # save_predictions
        image = self.generate_image()
        image2 = self.produce_entire_img()
        image2 = image2*127.5+127.5
        image = image*127.5+127.5
        Image.fromarray(image.astype(np.uint8)).save(output_folder+  \
            self.name + '/singles_' + str(epoch) +".png")
        Image.fromarray(image2.astype(np.uint8)).save(output_folder+  \
            self.name + '/entire_' + str(epoch) +".png")
        
        save_len = 100
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
    
    
    def lower_learning_rate(self):
    	decay= .9
    	K.set_value(model.optimizer.lr, decay * K.get_value(model.optimizer.lr))
    	print('Lowering learning rate to : ' + str(K.get_value(model.optimizer.lr)))
    	
    def load_best(self):
    	weight_file = output_folder + self.name + '/best.h5'
    	self.model.load_weights(weight_file)
        	
    def save_best(self,name='best'):
    	hf.save_model(self.model,self.name +'/' + name,folder=output_folder)
    	

    def set_inputs(self,inputs,input_img):
        self.inputs=inputs
        self.input_img = input_img

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

    def produce_entire_img(self):
        in_shape = self.model.input_shape
        out_shape = self.model.output_shape
        total_len = self.input_img.shape[0]
        entire_img = np.zeros(self.input_img.shape)
        entire_img[:,:] = -1
        sy, ey = 0,in_shape[1]
        dif = int((in_shape[1]-out_shape[1])/2)
        while(ey <= total_len):
            sx,ex = 0, in_shape[1]
            while(ex <= total_len):
                prediction = self.model.predict(self.input_img[sx:ex,sy:ey].reshape((1,)+in_shape[1:]), verbose=0)
                entire_img[sx+dif:ex-dif,sy+dif:ey-dif] = prediction.reshape(prediction.shape[1:3])
                ex+=out_shape[1]
                sx+=out_shape[1]
            ey+=out_shape[1]
            sy+=out_shape[1]

        return entire_img


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
    parser.add_argument("--normalize",type=bool,default=True)
    args = parser.parse_args()
    return args

args = get_args()
print('args: ' + str(args))

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
	# Things to explore: (1) dimensions of filters.  (2) activation function. (

	## Load model
	model = cg.projection_network(leaky_relu=lrelu[i],alpha=alpha[i],k1=f_dim1[i],k2=f_dim2[i],k3=f_dim3[i],nb_filters=nb_filters[i],elu=elu[i],longer=args.longer)
	model.name = 'deartifact_'+str(lrelu[i])+'_'+'{0:.3f}'.format(alpha[i])+ '_' + str(f_dim1[i]) + \
			'_' + str(f_dim2[i]) + '_' + str(f_dim3[i])+ '_' + str(nb_filters[i])+ \
			'_' + str(elu[i]) + '_' + str(lr[i])+ '_' + args.loss+ '_' + str(args.longer)+ \
			'_' + str(args.nb_epochs) + '_' + str(args.normalize)+ '_feb5'
	print('model name: ' + model.name)
	adam = Adam(lr=args.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	model.compile(loss=args.loss, optimizer=adam)

	## Load dataset
	X_train,X_test,Y_train,Y_test = hf.load_data(version=6,normalize_projection=args.normalize,\
				normalize=False)
				
	# X_train = X_train[1:100,:,:,:]
# 	Y_train = Y_train[1:100,:,:,:]

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

	## input image we are de artifacting
	H,g,f_true,f_recon = hf.load_H_g_target(version=2,index=90000)
	input_img = np.reshape(f_recon,(256,256))
	input_img = (input_img-np.min(input_img))/(np.max(input_img)-np.min(input_img))*255
	hf.generate_folder(output_folder + model.name)
	print('Writing to: projection_results/' + model.name)
	from PIL import Image
	Image.fromarray(input_img.astype(np.uint8)).save(output_folder+  \
		model.name + "/recon_input.png")
	targ_img = np.reshape(f_true,(256,256))
	targ_img = (targ_img-np.min(targ_img))/(np.max(targ_img)-np.min(targ_img))*255
	Image.fromarray(targ_img.astype(np.uint8)).save(output_folder+  \
		model.name + "/true_target.png")

	## Create callback
	if args.normalize:
		input_img = ((input_img-np.min(input_img))/(np.max(input_img)-np.min(input_img)))*2-1
	
	cb = generateImageCallback()
	cb.set_inputs(X_test[0:40,:,:,:],input_img)
	cb.set_name(model.name)

	##  Run model
	# X_train = X_train[1:200,:,:,:]
	# Y_train = Y_train[1:200,:,:,:]
	
	val_loss =  hf.run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=args.nb_epochs,batch_size=args.batch_size,DA=False,callback=cb,save_every=False,verbose=args.verbose)
	hf.save_model(model,model.name)
	



##
















# EoF #
