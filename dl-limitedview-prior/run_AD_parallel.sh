#!/bin/bash


# Train CNN
# python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --equal_AD True 






for i in `seq 1 29`; do
	# run generate_AD_dataset
	echo item: $i
	nohup python3 generate_AD_dataset.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --equal_AD True --tmp_save $i 2>&1&
done

python3 generate_AD_dataset.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --equal_AD True --tmp_save 30
sleep 14400
python3 combine_AD_dataset.py


export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# Train CNN again
python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --equal_AD True --use_combine True --num_gpus 8 --verbose 1


