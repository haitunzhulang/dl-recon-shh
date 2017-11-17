########## Train Projection CNN, generating outputs every epoch or so

# May 8 v41 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 41 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&

# May 9 v42 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&

# June 5 v4 0-4
# nohup python3 train7_resnet.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 4_0 --nb_epochs 4000 --num_gpus 2 --min_lr .001 2>&1&
# nohup python3 train7_resnet.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 4_1 --nb_epochs 4000 --num_gpus 2 --min_lr .001 2>&1&
# nohup python3 train7_resnet.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 4_2 --nb_epochs 4000 --num_gpus 2 --min_lr .001 2>&1&
# nohup python3 train7_resnet.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 4_3 --nb_epochs 4000 --num_gpus 2 --min_lr .001 2>&1&


# nohup python3 train7_resnet.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 4_4 --nb_epochs 4000 --num_gpus 2 --min_lr .001 2>&1&

from keras.callbacks import *
import helper_functions as hf
import CNN_generator as cg
import numpy as np
import argparse
import random
from keras.optimizers import SGD,Adam
import contextlib
import os
import math
import sys

from keras import backend as K
from multi_gpu import make_parallel

output_folder = 'projection_results7/'
hf.generate_folder(output_folder)
hf.generate_folder('models/')

# import keras.backend.tensorflow_backend as KTF
# import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# KTF.set_session(get_session())


class generateImageCallback(Callback):
	def on_train_begin(self, logs={}):
		self.losses = []
		self.val_losses = []
		self.lrs = []
	

	def on_epoch_end(self, epoch,logs={}):
		self.losses.append(logs.get('loss'))
		self.val_losses.append(logs.get('val_loss'))
		self.lrs.append(K.get_value(model.optimizer.lr))
		epoch += self.num_epochs

	
		save_iter=45
		save_len = 5
		print(logs)
		if epoch<10:
			self.save_best()
			self.best_loss = logs['loss']
		elif logs['loss'] < self.best_loss and epoch%save_iter==0: # improvement, save it!
			self.save_best()
			self.best_loss=logs['loss']
		elif logs['loss']>2*self.best_loss and epoch < 10000: ## loss has exploded!
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
		self.model_simple.set_weights(self.model.get_weights())
		hf.save_model(self.model_simple,self.name +'/' + name+'_simple',folder=output_folder)

	def set_simple_cnn(self,model_simple):
		self.model_simple = model_simple
	
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
	parser.add_argument("--nb_filters", type=int, default=64)
	parser.add_argument("--nb_epochs", type=int, default=1000)
	parser.add_argument("--lr",type=float,default=.001)
	parser.add_argument("--loss",type=str,default='mse')
	parser.add_argument("--verbose",type=int,default=1)
	parser.add_argument("--batch_size",type=int,default=64)
	parser.add_argument("--dropout",type=float,default=0.0)
	parser.add_argument("--dataset",type=str,default='4_0')
	parser.add_argument("--pretrained_model",type=str,default=None)
	parser.add_argument("--num_gpus",type=int,default=1)
	parser.add_argument("--depth",type=int,default=20)
	parser.add_argument("--AD",type=bool,default=False)
	parser.add_argument("--dimension",type=int,default=256) # probably never change
	parser.add_argument("--just_return_name",type=bool,default=False)
	parser.add_argument("--name",type=str,default='noname')
	args = parser.parse_args()
	return args

args = get_args()
print(args)


def get_model_name(args=args):
	name = 'deartifact_' + \
		'_lr' + str(args.lr)+ '_' + args.loss+  \
		'_numEpochs_' + str(args.nb_epochs) + '_DO'+ str(args.dropout) + \
		'_v' + str(args.dataset)+ \
		'_batchsize_' + str(args.batch_size) + \
		'_depth_'+ str(args.depth) + \
		'_AD_' + str(args.AD) + \
		'_nbfilters_' + str(args.nb_filters) + \
		'_name_' + args.name
	return name

input_shape = (args.dimension,args.dimension,1)

model = cg.residual_projectionNet2(depth=args.depth,nb_filters=args.nb_filters,input_shape=input_shape,dropout=args.dropout)

if args.num_gpus > 1: 
	model = make_parallel(model,args.num_gpus)
	args.batch_size = args.batch_size*args.num_gpus

model.name = get_model_name()

if args.pretrained_model is not None:
	best = hf.load_trained_CNN(name=args.pretrained_model,folder='')
	model.set_weights(best.get_weights())


print('model name: ' + model.name)
print('#######checking ...#######')
if args.just_return_name:
	with open('/home/shenghua/DL-recon/dl-limitedview-prior/tmp.out', 'w') as outfile:
		outfile.write(get_model_name())
	print('#######system exits#######')	
	sys.exit()

print('#####continue#######')
opt = Adam(lr=args.lr)
#opt = SGD(lr=args.min_lr, momentum=0.9, decay=0.0001, nesterov=True)
loss_weights = []
loss_weights.append(1.0)


model.compile(loss=args.loss, optimizer=opt,loss_weights=loss_weights)


hf.generate_folder(output_folder+model.name)


cb = generateImageCallback()

cb.set_training_nums_suffix('')
#     cb.set_inputs(X_test[0:8,:,:,:])
cb.set_name(model.name)


cb.set_num_epochs(0)
model_simple = cg.residual_projectionNet2(depth=args.depth,nb_filters=args.nb_filters,input_shape=input_shape,dropout=args.dropout)

cb.set_simple_cnn(model_simple)

cbs = []
cbs.append(cb)


val_loss = hf.run_model_generator(model,dataset=args.dataset,nb_epoch=args.nb_epochs,batch_size=args.batch_size,callbacks=cbs,verbose=args.verbose,AD=args.AD,num_gpus=args.num_gpus)



K.clear_session()







# EoF #
