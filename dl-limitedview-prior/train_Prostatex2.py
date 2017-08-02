
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

output_folder = 'Prostatex2_Results/'
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
		epoch += self.num_epochs

		save_len = 5
		if epoch%save_len==0:
			self.save_best(name=str(epoch))
		
		# Save training plot
		hf.save_training_plot(self.losses,self.val_losses,self.lrs,model_name=self.model.name,plot_folder=output_folder,suffix=self.training_nums_suffix)
		np.savetxt(output_folder + self.model.name + '/training_nums' +self.training_nums_suffix + '.out', (self.losses,self.val_losses,self.lrs), delimiter=',')
	
	def set_training_nums_suffix(self,x):
		self.training_nums_suffix=str(x)

	
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


def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--f_dim1", type=int, default=16)
	parser.add_argument("--f_dim2", type=int, default=1)
	parser.add_argument("--f_dim3", type=int, default=8)
	parser.add_argument("--nb_filters", type=int, default=512)
	parser.add_argument("--nb_epochs", type=int, default=100)
	parser.add_argument("--lr",type=float,default=.001)
	parser.add_argument("--loss",type=str,default='mse')
	parser.add_argument("--verbose",type=int,default=1)
	parser.add_argument("--batch_size",type=int,default=256)
	parser.add_argument("--dropout",type=float,default=0.0)
	parser.add_argument("--dataset",type=str,default='4_0')
	parser.add_argument("--pretrained_model",type=str,default=None)
	parser.add_argument("--num_normal_training_examples",type=int,default=1000)
	parser.add_argument("--num_gpus",type=int,default=1)
	parser.add_argument("--use_combine",type=bool,default=False)
	parser.add_argument("--best_is_old",type=bool,default=False)
	parser.add_argument("--combine_folder",type=str,default='tmp/')
	parser.add_argument("--use_cycling_lr",type=bool,default=False)
	parser.add_argument("--min_lr",type=float,default=.0001)
	parser.add_argument("--max_lr",type=float,default=.005)
	parser.add_argument("--lr_ss",type=int,default=5)
	parser.add_argument("--regularizer_loss_weight",type=float,default=0.)
	parser.add_argument("--depth",type=int,default=20)
	parser.add_argument("--AD",type=bool,default=False)
	parser.add_argument("--dimension",type=int,default=256)
	parser.add_argument("--just_return_name",type=bool,default=False)
	parser.add_argument("--name",type=str,default='noname')
	args = parser.parse_args()
	return args

