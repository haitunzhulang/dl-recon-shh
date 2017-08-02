
# This function will show a make a selection of calls to train several CNNs

######### Experiment 1 - Acc vs Params ############

#### Need to train some number of models, lets say 20 models, where each has a different number of hyper parameters.
#### We can vary this number of hyper parameters by changing the nb_filters in each layer or by changing depth, or both.

import math
import os

name = 'Experiment_1'
nb = 6
nb_filters = []
while nb <= 66:
	nb_filters.append(nb)
	nb+=6

dataset = '4_0'
depth = 10
move_dataset = '0'


SAFE = True

for i in range(len(nb_filters)):
	which_gpu = math.floor((i+1)/4.)
	which_name = name + '_' + str(i)
	if SAFE:
		print( 'python3 one_button_training.py --nb_filters ' + str(nb_filters[i]) + ' --dataset ' + dataset + ' --depth ' + str(depth) + ' --which_gpu ' + str(which_gpu) + ' --name ' + which_name + ' --move_dataset ' + move_dataset)
	else:
		os.system('nohup python3 one_button_training.py --nb_filters ' + str(nb_filters[i]) + ' --dataset ' + dataset + ' --depth ' + str(depth) + ' --which_gpu ' + str(which_gpu) + ' --name ' + which_name + ' --move_dataset ' + move_dataset + ' 2>&1&')
	

for i in range(len(nb_filters)):
	which_gpu = math.floor((i+1)/4.)
	which_name = name + '_' + str(i)
	print('nohup python3 one_button_training.py --nb_filters ' + str(nb_filters[i]) + ' --dataset ' + dataset + ' --depth ' + str(depth) + ' --which_gpu ' + str(which_gpu) + ' --name ' + which_name + ' --move_dataset ' + move_dataset + ' 2>&1&')


# This configuration of gpus worked for this setup
# nohup python3 one_button_training.py --nb_filters 6 --dataset 4_0 --depth 10 --which_gpu 0 --name Experiment_1_0 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 12 --dataset 4_0 --depth 10 --which_gpu 0 --name Experiment_1_1 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 18 --dataset 4_0 --depth 10 --which_gpu 1 --name Experiment_1_2 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 24 --dataset 4_0 --depth 10 --which_gpu 3 --name Experiment_1_3 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 30 --dataset 4_0 --depth 10 --which_gpu 1 --name Experiment_1_4 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 36 --dataset 4_0 --depth 10 --which_gpu 3 --name Experiment_1_5 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 42 --dataset 4_0 --depth 10 --which_gpu 3 --name Experiment_1_6 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 48 --dataset 4_0 --depth 10 --which_gpu 2 --name Experiment_1_7 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 54 --dataset 4_0 --depth 10 --which_gpu 2 --name Experiment_1_8 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 60 --dataset 4_0 --depth 10 --which_gpu 1 --name Experiment_1_9 --move_dataset False 2>&1&
# nohup python3 one_button_training.py --nb_filters 66 --dataset 4_0 --depth 10 --which_gpu 0 --name Experiment_1_10 --move_dataset False 2>&1&


######### Experiment 2 - Acc vs Number of Training Stages ############

name = 'Experiment_2_'
dataset = '4_0'
depth = 10
nb_filters=18
which_gpu = 1
move_dataset='0'
batch_size = 64
num_stages = 10
print('python3 one_button_training.py --nb_filters ' + str(nb_filters) + \
	' --dataset ' + dataset + ' --depth ' + str(depth) + ' --which_gpu ' + str(which_gpu) + \
	' --name ' + name + ' --move_dataset ' + move_dataset + ' --num_stages ' + str(num_stages) + \
	' --batch_size ' + str(batch_size) )
print('nohup python3 one_button_training.py --nb_filters ' + str(nb_filters) + \
	' --dataset ' + dataset + ' --depth ' + str(depth) + ' --which_gpu ' + str(which_gpu) + \
	' --name ' + name + ' --move_dataset ' + move_dataset + ' --num_stages ' + str(num_stages) + \
	' --batch_size ' + str(batch_size) + ' 2>&1&')

# 5534 - jun 19 9:39pm
# python3 one_button_training.py --nb_filters 6 --dataset 4_0 --depth 10 --which_gpu 0 --name Experiment_2_6_ --move_dataset 0 --num_stages 10 --batch_size 64 --skipping_first_iteration 1
#  
# python3 one_button_training.py --nb_filters 12 --dataset 4_0 --depth 10 --which_gpu 1 --name Experiment_2_12_ --move_dataset 0 --num_stages 10 --batch_size 64
#  
# python3 one_button_training.py --nb_filters 18 --dataset 4_0 --depth 10 --which_gpu 2 --name Experiment_2_18_ --move_dataset 0 --num_stages 10 --batch_size 64
#  
# python3 one_button_training.py --nb_filters 24 --dataset 4_0 --depth 10 --which_gpu 3 --name Experiment_2_24_ --move_dataset 0 --num_stages 10 --batch_size 64


######### Experiment 3 - Acc vs Number of Training Stages FOR TVC Data ############
# python3 one_button_training.py --nb_filters 24 --dataset 5 --depth 10 --which_gpu 1 --name Experiment_3_0_ --move_dataset 1 --num_stages 5 --batch_size 64



######### Experiment 4 - Acc vs Redoing EXP 3 but for TVC Data!! ############
# python3 one_button_training.py --nb_filters 6 --dataset 6 --depth 10 --which_gpu 2 --name Experiment_4_6_TEST_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 12 --dataset 6 --depth 10 --which_gpu 1 --name Experiment_4_12_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 18 --dataset 6 --depth 10 --which_gpu 2 --name Experiment_4_18_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 24 --dataset 6 --depth 10 --which_gpu 3 --name Experiment_4_24_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 30 --dataset 6 --depth 10 --which_gpu 3 --name Experiment_4_30_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 36 --dataset 6 --depth 10 --which_gpu 2 --name Experiment_4_36_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 42 --dataset 6 --depth 10 --which_gpu 1 --name Experiment_4_42_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000
# python3 one_button_training.py --nb_filters 48 --dataset 6 --depth 10 --which_gpu 0 --name Experiment_4_48_ --move_dataset 0 --num_stages 4 --batch_size 32 --num_AD 1000




# EoF #