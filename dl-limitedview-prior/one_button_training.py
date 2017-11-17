import os
import sys
import csv
import numpy as np
import helper_functions as hf
from shutil import copyfile
import argparse
import tensorflow as tf
import glob
import time




def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--nb_filters", type=int, default=16)
	parser.add_argument("--dataset",type=str,default='4_0')
	parser.add_argument("--depth",type=int,default=20)
	parser.add_argument("--dimension",type=int,default=256)
	parser.add_argument("--num_AD",type=int,default=10000)
	parser.add_argument("--which_gpu",type=int,default=0)
	parser.add_argument("--num_gpus",type=int,default=1)
	parser.add_argument("--num_stages",type=int,default=5)
	parser.add_argument("--nb_epochs",type=int,default=40)
	parser.add_argument("--name",type=str,default='third')
	parser.add_argument("--batch_size",type=int,default=16)
	parser.add_argument("--move_dataset",type=int,default=1)
	parser.add_argument("--skipping_first_iteration",type=int,default=0)
	parser.add_argument("--TVConstraint",type=int,default=0)
	parser.add_argument("--start_index_for_loop",type=int,default=0)
	parser.add_argument("--non_generator",type=int,default=0)
	parser.add_argument("--train_on_chpc",type=int,default=0)
	args = parser.parse_args()
	return args




args = get_args()
print(args)

# print(args.move_dataset)
# sys.exit()

# Prepare different starting datasets
dataset_dict,theta_dict = {}, {}
dataset_dict['4_0'] = '/home/shenghua/DL-recon/xct-parallelbeam-matlab/dataset_v4_50D_Nonneg_NonIC_Noise_0/'
theta_dict['4_0'] = 50
dataset_dict['5'] = '/home/bmkelly/xct-parallelbeam-matlab/dataset_v5_50D_IC_Noise_TVC/'
theta_dict['5'] = 50
dataset_dict['6'] = '/home/shenghua/DL-recon/xct-parallelbeam-matlab/dataset_v5_100D_noIC_Noise/'
dataset_dict['7_1k'] = '/home/bmkelly/xct-parallelbeam-matlab/dataset_v7_80D_IC_Noise_TVC/'
theta_dict['6'] = 100
theta_dict['7_1k'] = 80

nb_filters = args.nb_filters
depth = args.depth
dataset = args.dataset
dimension = args.dimension
dataset_name = dataset_dict[dataset].split('/')[-2]
num_AD = args.num_AD
num_gpus = args.num_gpus
which_gpu=args.which_gpu
num_stages=args.num_stages
nb_epochs = args.nb_epochs
theta=theta_dict[dataset]
if num_gpus==2:
#	os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu) + ',' + str(which_gpu+1)+ ',' + str(which_gpu+2)+ ',' + str(which_gpu+3)
	os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu+2) + ',' + str(which_gpu+3)
	print('Env: ' + os.environ["CUDA_VISIBLE_DEVICES"])
if num_gpus==1:
	which_gpu=0
	os.environ["CUDA_VISIBLE_DEVICES"] = str(which_gpu)
name = args.name
skipping_first_iteration = args.skipping_first_iteration
num_per_worker = 10

batch_size = args.batch_size

range_i = []
for i in range(num_stages):
	range_i.append(i)

# folder for this training run:
def send_message(message):
	print('********************************************************')
	print(message)
	print('********************************************************')


for i in range_i[args.start_index_for_loop:]:
	print('------------- stage '+str(i)+' begins...-----------------------')
	name_of_run = name
	stage_name = name+str(i)
	mf = 'training_runs/' + name_of_run + '/'

	save_dir = stage_name + '_AD/'
	

	#### Train model ####
	# Set number of GPUS:
	if i ==0:
		### Generate new dataset with AD_files
		send_message('Generating Initial Training Dataset')
		### Generate new dataset with AD_files
		DIRNAME = dataset_dict[dataset]
		trainf = 'datasets/v' + dataset+ '_train/'
		testf = 'datasets/v' + dataset+ '_test/'
		testFinalf = 'datasets/v' + dataset+ '_testFinal/'
	
		ntrain=7500
		ntest=1000
		ntestFinal = 500
# 		AD_path = AD_folder
		hf.make_dataset(trainf,testf,testFinalf,ntrain,ntest,ntestFinal,DIRNAME,AD_path=None,num_std_away=100)
		
		train_command = 'python3 train7_resnet.py --nb_filters ' + str(nb_filters)+  \
			' --depth ' + str(depth) + ' --batch_size ' + str(batch_size) + ' --dataset ' + dataset + ' --nb_epochs ' + str(nb_epochs) +\
			' --name ' + stage_name + ' --num_gpus ' + str(args.num_gpus)
	else:
		train_command = 'python3 train7_resnet.py --nb_filters ' + str(nb_filters)+  \
			' --depth ' + str(depth) + ' --batch_size ' + str(batch_size) + ' --dataset ' + name + str(i-1) + ' --nb_epochs ' + str(nb_epochs) +\
			' --name ' + stage_name + ' --AD True' + ' --num_gpus ' + str(args.num_gpus)
	
	# train
	pretrained_model_addition = ''
	if i>0:
		pretrained_model_addition = ' --pretrained_model ' + mf + 'stage' + str(i-1) + '_model/best_simple'

	if not (skipping_first_iteration!=0 and i==range_i[args.start_index_for_loop]):
		print('#######pretraining#######')
		print((skipping_first_iteration,i))
		send_message('Attempting to train with this command: ' + train_command + pretrained_model_addition)
		os.system(train_command + pretrained_model_addition)


	#### Find and move best model to transfer over ####
	# Saving/Moving best model, a new training plot/data, visualization of filters
	model_folder = mf + 'stage' + str(i) + '_model/'
#	os.system(train_command)
	os.system(train_command + ' --just_return_name True')
 
	with open('/home/shenghua/DL-recon/dl-limitedview-prior/tmp.out','r') as infile:
		model_name = infile.readline()

	new_folder = model_folder
	hf.generate_folder(new_folder)
	old_folder = 'projection_results7/' + model_name + '/'

	print('------------- stage '+str(i)+' training ends!!-----------------------')
	# Eventual AD directory on turing
	AD_folder = 'datasets/v' + save_dir


	### Generate .batch file to train this model
	def generate_run_file(model_folder_func='models/' + stage_name + '/',nb_filters=nb_filters,depth=depth,file_name=stage_name,new_folder=new_folder,dataset_name=dataset_name,save_dir=AD_folder,AD_folder=AD_folder,num_AD=num_AD,theta=theta):

		num_per_worker = 10
		run_file_name = new_folder + stage_name + '.batch'

		with open(run_file_name, 'w') as f:
			f.write('#!/bin/bash\n') # This has to be here
			f.write('#PBS -l nodes=1:ppn=1,walltime=03:59:00\n') # resources
			f.write('#PBS -N ' + file_name +'\n') # name of run
			#f.write('#PBS -m be\n') # Email when the job begins and ends 
			f.write('#PBS -e log -o log\n') # Store PBS log files in a separate directory 
			f.write('#PBS -t 0-' + str(int(num_AD/num_per_worker)) + '\n') # Request an array of worker
			f.write('\n')
			f.write('IMG_NAME="singularity-ubuntu16.04-tf1.0-gpu-python3.img"\n')
			f.write('cd /scratch/shenghuahe \n')
			f.write('module load singularity\n')
			f.write('module load cuda-8.0\n')
			f.write('echo "PBS_ARRAYID:" $PBS_ARRAYID\n')
			f.write('singularity exec -B /scratch:/scratch $IMG_NAME python3 generate_AD_dataset.py --dataset ' \
			 + dataset_name + ' --depth ' + str(depth) + ' --nb_filters ' + str(nb_filters) + ' --model_folder ' + model_folder_func + \
			  ' --save_dir ' + save_dir + ' --num_add ' + str(num_per_worker) + ' --theta ' + str(theta) + \
			  ' --residualNet True --tmp_save $PBS_ARRAYID' + ' --TVConstraint ' + str(args.TVConstraint) + '\n')
	
			f.write('\n')
# 			f.write('scp ' + save_dir + '$PBS_ARRAYID.pkl bmkelly@turing.seas.wustl.edu:/home/bmkelly/dl-limitedview-prior/' + AD_folder)
		return run_file_name


	def parse_training_nums(file_name):
		file = open(file_name, 'r') 
		train_loss = file.readline().split(',')
		val_loss = file.readline().split(',')
		learning_rate = file.readline().split(',')

		train_loss = [float(i) for i in train_loss]
		val_loss = [float(i) for i in val_loss]
		learning_rate = [float(i) for i in learning_rate]
		return train_loss,val_loss,learning_rate

	def move_and_save(this_run=i,new_folder=new_folder,old_folder=old_folder,dataset_folder=dataset_dict[dataset],stage_name=stage_name,AD_folder=AD_folder,move_dataset=args.move_dataset):
		### Move dataset over?
		if move_dataset==1 and this_run==0:
			send_message('Moving dataset over: ' + dataset_folder)
			os.system('scp -r ' + dataset_folder + ' shenghuahe@dtn01.chpc.wustl.edu:/scratch/shenghuahe/datasets/')
	
		send_message('Moving model files around, in order to ship over to chpc')
		# Locate important files:
		training_nums = old_folder + 'training_nums.out'
		# parse this file:
		train_loss,val_loss,learning_rate = parse_training_nums(training_nums)

		# Find the lowest val_loss
		min_index = np.argmin(val_loss)
		min_val_loss = val_loss[min_index]

		# Find the lowest train_loss at the lowest val_loss
		min_val_train_loss = train_loss[min_index]

		# Find lowest overall train_loss
		min_train_loss = np.min(train_loss)
		
		best_index = 'best'

		### ? Create visualization of filters for this specific model
		model = hf.load_trained_CNN(version=1,name=best_index + '_simple',folder=old_folder)

		### Move model file into transfer folder
		model_file_yaml = old_folder + best_index + '_simple.yaml'
		model_file_h5 = old_folder + best_index + '_simple.h5'
		copyfile(model_file_yaml, new_folder + 'best_simple.yaml')
		copyfile(model_file_h5, new_folder + 'best_simple.h5')
		
		### Move training_nums file over
		copyfile(training_nums,new_folder + 'training_nums.out')
		
		### Move png plot over
		copyfile(old_folder + 'training_plot.png', new_folder + 'training_plot.png')

		### Move .batch file over
		run_file_name = generate_run_file()
		os.system('scp ' + run_file_name + ' shenghuahe@dtn01.chpc.wustl.edu:/scratch/shenghuahe/batch_files')

		### Move model folder over
		os.system('scp -r ' + new_folder + ' shenghuahe@dtn01.chpc.wustl.edu:/scratch/shenghuahe/models/' + stage_name)

		### Make directory for the AD files to be transfered into:
		hf.generate_folder(AD_folder)

		###  Create AD folder on CHPC to put files into
		os.system('ssh shenghuahe@dtn01.chpc.wustl.edu "mkdir /scratch/shenghuahe/' + AD_folder + '"')

		return model



	### Move stuff over function call ###	
	if not (skipping_first_iteration!=0 and i==range_i[args.start_index_for_loop]):
		model = move_and_save()
	
		
		send_message('Calling batchfile on chpc')
		### Call batchfile
		os.system('ssh shenghuahe@dtn01.chpc.wustl.edu "qsub /scratch/shenghuahe/batch_files/'+stage_name+'.batch"')

	# if not (skipping_first_iteration!=0 and i==0):
# 		send_message('Generating Experimental results')
# 		# Generate experimental results
# 		start_index=7501
# 		num_val=100
# 		Verbose=0
# 		for i in range(num_val):
# 			index = start_index+i
# 			fname = stage_name+str(index)
# 			hf.compare_psnr(model,njumps=15,mindex=-1,mf='projection_results7/',niters=10000,cutoff=.001,title=fname,index=index,version=3,theta=theta,display=False,pls=True,data_dirname=dataset_dict[dataset],PSNR=True,linked=False,Verbose=0)

# if 1==1 or ...: - 
	if not (skipping_first_iteration !=0 and i==range_i[args.start_index_for_loop]):
		if not i==0:
			last_AD_folder='datasets/v'+name+str(i-1)+'_AD/'
			os.system('rm -rf ' + last_AD_folder[0:-1])
			last_AD_folder='datasets/v'+name+str(i-1)+'_train_AD/'
			os.system('rm -rf ' + last_AD_folder[0:-1])
		os.system('scp -r shenghuahe@dtn01.chpc.wustl.edu:/scratch/shenghuahe/' + AD_folder[0:-1] + ' datasets/')
		# If we are training for another round:
		send_message('Waiting for AD folder to contain ' + str(num_AD))
		### Wait until AD_folder contains 900 files, or close to 900 files
		sleep_time = 3600*2
		# num_AD = 10000
	
		while len(glob.glob(AD_folder + '*.pkl')) < (num_AD/num_per_worker) - 10:
			print('Sleeping for ' + str(sleep_time) + ' seconds.  Len directory: ' + str(len(glob.glob(AD_folder + '*.pkl'))) +'. Directory: ' + AD_folder)
			time.sleep(sleep_time)  # Delay for hour
			os.system('scp -r shenghuahe@dtn01.chpc.wustl.edu:/scratch/shenghuahe/' + AD_folder[0:-1] + ' datasets/')
			
		send_message('Generating New Dataset')
		### Generate new dataset with AD_files
		DIRNAME = dataset_dict[dataset]
		trainf = 'datasets/v' + stage_name+ '_train_AD/'
		testf = 'datasets/v' + stage_name+ '_test_AD/'
		testFinalf = 'datasets/v' + stage_name+ '_testFinal_AD/'
	
		ntrain=7500
		ntest=1000
		ntestFinal = 500
		AD_path = AD_folder
		hf.make_dataset(trainf,testf,testFinalf,ntrain,ntest,ntestFinal,DIRNAME,AD_path=AD_path,num_std_away=100)




























# EoF #
