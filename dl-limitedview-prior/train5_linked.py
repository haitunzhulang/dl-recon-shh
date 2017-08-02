########## Train Projection CNN, generating outputs every epoch or so

# This is the same as train3_projection_cnn.py except we are stacking the CNN once and linking the
# weights.


# Mar 13 -- linked CNN idea
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 100 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True 2>&1&

#smaller init, more AD examples
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 2>&1&

# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 100 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 2>&1&

# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 100 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 3 2>&1&

# Just running to generate 10k training images 
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 1 --inter_epochs 50 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 10000 2>&1&


# (1) AD Training >> normal training
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 10 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --equal_AD True 2>&1&

# mar 15 -- deeper CNN
# nohup python3 train5_linked.py --nb_filters=512 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 20 --final_act True --dataset 18 --init_epochs 500 --inter_epochs 20 --num_loop 5 --lr .0005 --DA_Lr_decrease 1 --num_add 1000 --num_stacks 1 --equal_AD True --num_gpus 2 2>&1&

# mar16 AD Training >> normal training
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --final_act True --dataset 13 --init_epochs 2 --inter_epochs 20 --num_loop 5 --lr .0001 --DA_Lr_decrease 1 --use_previous_best True --num_add 1000 --num_stacks 1 --use_combine True --num_gpus 8 2>&1&

# mar20 Normal training
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --dataset 19 --init_epochs 400 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --use_previous_best True  --num_stacks 1 --num_gpus 8 2>&1&
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --dataset 21 --init_epochs 600 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --use_previous_best True  --num_stacks 1 --num_gpus 4 2>&1&
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --dataset 22 --init_epochs 600 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --use_previous_best True  --num_stacks 1 --num_gpus 4 2>&1&

# mar 23 v19 - with AD Combined
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 30 --dataset 19 --init_epochs 400 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --use_previous_best True  --num_stacks 1 --use_combine True --combine_folder v19_AD --num_gpus 1 2>&1&

# mar 23 v13 -- original, no AD
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 13 --init_epochs 500 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --num_stacks 1 --num_gpus 8 2>&1&

# mar 24 v19 -- AD
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 19 --init_epochs 500 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --num_stacks 1 --use_combine True --combine_folder v19_AD --use_previous_best True --num_gpus 1 2>&1&

# mar 24 v23 -- AD
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 23 --init_epochs 1000 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --num_stacks 1 --use_combine True --combine_folder v23_AD --use_previous_best True --num_gpus 8 2>&1&


# mar 26 -- AD
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 25 --init_epochs 1000 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --num_stacks 1  --num_gpus 7 2>&1&

# mar 28 - AD, big AD, for v23
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 23 --init_epochs 1000 --inter_epochs 0  --lr .0001 --DA_Lr_decrease 1 --num_stacks 1 --use_combine True --combine_folder v23_AD --use_previous_best True --num_gpus 8 2>&1&

# apr 4 - AD huge dataset - fit_generator v23
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 23 --init_epochs 1000 --use_previous_best True --num_gpus 8 --use_cycling_lr True --min_lr .00005 --max_lr .0005 --lr_ss 8 --verbose 1 2>&1&

# testing with regularizer on output of CNN  apr4
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 28 --dataset 23 --init_epochs 4000 --use_previous_best True --num_gpus 1 --use_cycling_lr True --min_lr .00005 --max_lr .0005 --lr_ss 40 --verbose 1 --regularizer_loss_weight .00001 2>&1&
# focused testing of regularizer on output of CNN apr 4
# nohup python3 train5_linked.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 23 --init_epochs 4000 --num_gpus 4 --use_cycling_lr True --min_lr .0001 --max_lr .001 --lr_ss 40 --regularizer_loss_weight .0001 --verbose 1 2>&1&
# nohup python3 train5_linked.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 23 --init_epochs 4000 --num_gpus 4 --use_cycling_lr True --min_lr .0001 --max_lr .001 --lr_ss 40 --verbose 1 2>&1&


# apr 6 - v25 Noisy
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 4 --dataset 25 --init_epochs 4000 --num_gpus 8 --use_cycling_lr True --min_lr .0000005 --max_lr .0005 --lr_ss 120 --verbose 1 2>&1&

# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 25 --init_epochs 4000 --num_gpus 8 --use_cycling_lr True --min_lr .00001 --max_lr .1 --lr_ss 120 --residualNet True --verbose 1 2>&1&

# nohup python3 train5_linked.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 27 --init_epochs 4000 --num_gpus 8 --residualNet True --depth 20 --verbose 1 2>&1&

# apr 9 - tv+noise v27
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 27 --init_epochs 4000 --num_gpus 8 --verbose 1 2>&1&
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 19 --init_epochs 4000 --num_gpus 8 --min_lr .001 --residualNet True --use_previous_best True --verbose 1 2>&1&

# apr 10 - retrain v19 -- D100
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 19 --init_epochs 4000 --num_gpus 8 --min_lr .001 --residualNet True --verbose 1 2>&1&
# python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 19 --init_epochs 4000 --num_gpus 8 --min_lr .000101 --residualNet True --verbose 1

# Apr 11 - train v25 AD!
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 25 --init_epochs 4000 --num_gpus 8 --min_lr .001 --residualNet True --use_previous_best True --verbose 1 2>&1&

# Apr 12 - train v19 AD!
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 19 --init_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --use_previous_best True 2>&1&

# Apr 13 - train v28 Reg
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 28 --init_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --verbose 1 2>&1&

# Apr 14 - train v29 Reg
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 29 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --verbose 1 2>&1&

# Apr 19 - Train v31 Reg
# nohup python3 train5_linked.py --nb_filters=325 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --batch_size 8 --dataset 31 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --verbose 1 2>&1&
# nohup python3 train5_linked.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --depth 20 --batch_size 8 --dataset 31 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --verbose 1 2>&1&

# Apr 21 -- Train v31 AD
# nohup python3 train5_linked.py --nb_filters=64 --f_dim1 16 --f_dim2 1 --f_dim3 8 --loss mse --depth 20 --batch_size 8 --dataset 31 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --AD True --use_previous_best True --verbose 1 2>&1&

# Apr 24 -- Train v312 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 8 --dataset 312 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --AD True --use_previous_best True --verbose 1 2>&1&
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 4 --batch_size 8 --dataset 312 --nb_epochs 4000 --num_gpus 8 --min_lr .0001 --residualNet True --AD True --verbose 1 2>&1&

# Apr 27 - Train v35 Reg
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 10 --dataset 35 --nb_epochs 4000 --num_gpus 6 --min_lr .0001 --residualNet True --verbose 1 2>&1&

# Apr 28 - Train v36 Reg
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 10 --dataset 36 --nb_epochs 4000 --num_gpus 4 --min_lr .0001 --residualNet True --verbose 1 2>&1&


# May 2 - Train v35 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 16 --dataset 35 --nb_epochs 4000 --num_gpus 4 --min_lr .0001 --residualNet True --AD True --use_previous_best True --verbose 1 2>&1&
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 16 --dataset 352 --nb_epochs 4000 --num_gpus 4 --min_lr .0001 --residualNet True --AD True --use_previous_best True --verbose 1 2>&1&

# May 2 - Train v37
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 64 --dataset 353 --nb_epochs 4000 --num_gpus 1 --min_lr .0001 --residualNet True --use_previous_best True --AD True --verbose 1 2>&1&

# May 4 - v38 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 16 --dataset 38 --nb_epochs 4000 --num_gpus 4 --min_lr .0001 --residualNet True --use_previous_best True --AD True --verbose 1 2>&1&

# May 5 v42, v39
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 39 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --verbose 1 2>&1&

# May 7 - v40/v41 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 40 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&

# May 8 v41 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 41 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&

# May 9 v42 AD
# nohup python3 train5_linked.py --nb_filters=64 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&

# June 5 v4 0-4
# nohup python3 train5_linked.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&
# nohup python3 train5_linked.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&
# nohup python3 train5_linked.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&
# nohup python3 train5_linked.py --nb_filters=16 --loss mse --depth 20 --batch_size 32 --dataset 42 --nb_epochs 4000 --num_gpus 2 --min_lr .0001 --residualNet True --use_previous_best True --AD True 2>&1&


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

from keras import backend as K
from multi_gpu import make_parallel



output_folder = 'projection_results7/'
hf.generate_folder(output_folder)
hf.generate_folder('models/')

class CyclicLR(Callback):
    """This callback implements a cyclical learning rate policy (CLR).
    The method cycles the learning rate between two boundaries with
    some constant frequency, as detailed in this paper (https://arxiv.org/abs/1506.01186).
    The amplitute of the cycle can be scaled on a per-iteration or 
    per-cycle basis.
    This class has three built-in policies, as put forth in the paper.
    "triangular":
        A basic triangular cycle w/ no amplitude scaling.
    "triangular2":
        A basic triangular cycle that scales initial amplitude by half each cycle.
    "exp_range":
        A cycle that scales initial amplitude by gamma**(cycle iterations) at each 
        cycle iteration.
    For more detail, please see paper.
    
    # Example
        ```python
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., mode='triangular')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```
    
    Class also supports custom scaling functions:
        ```python
            clr_fn = lambda x: 0.5*(1+np.sin(x*np.pi/2.))
            clr = CyclicLR(base_lr=0.001, max_lr=0.006,
                                step_size=2000., scale_fn=clr_fn,
                                scale_mode='cycle')
            model.fit(X_train, Y_train, callbacks=[clr])
        ```    
    # Arguments
        base_lr: initial learning rate which is the
            lower boundary in the cycle.
        max_lr: upper boundary in the cycle. Functionally,
            it defines the cycle amplitude (max_lr - base_lr).
            The lr at any cycle is the sum of base_lr
            and some scaling of the amplitude; therefore 
            max_lr may not actually be reached depending on
            scaling function.
        step_size: number of training iterations per
            half cycle. Authors suggest setting step_size
            2-8 x training iterations in epoch.
        mode: one of {triangular, triangular2, exp_range}.
            Default 'triangular'.
            Values correspond to policies detailed above.
            If scale_fn is not None, this argument is ignored.
        gamma: constant in 'exp_range' scaling function:
            gamma**(cycle iterations)
        scale_fn: Custom scaling policy defined by a single
            argument lambda function, where 
            0 <= scale_fn(x) <= 1 for all x >= 0.
            mode paramater is ignored 
        scale_mode: {'cycle', 'iterations'}.
            Defines whether scale_fn is evaluated on 
            cycle number or cycle iterations (training
            iterations since start of cycle). Default is 'cycle'.
    """

    def __init__(self, base_lr=0.001, max_lr=0.006, step_size=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size = step_size
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: 1/(2.**(x-1))
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma**(x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step_size=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step_size != None:
            self.step_size = new_step_size
        self.clr_iterations = 0.
        
    def clr(self):
        cycle = np.floor(1+self.clr_iterations/(2*self.step_size))
        x = np.abs(self.clr_iterations/self.step_size - 2*cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr-self.base_lr)*np.maximum(0, (1-x))*self.scale_fn(self.clr_iterations)
        
    def on_train_begin(self, logs={}):
        logs = logs or {}

        if self.clr_iterations == 0:
            K.set_value(self.model.optimizer.lr, self.base_lr)
        else:
            K.set_value(self.model.optimizer.lr, self.clr())        
            
    def on_batch_end(self, epoch, logs=None):
        
        logs = logs or {}
        self.trn_iterations += 1
        self.clr_iterations += 1
        K.set_value(self.model.optimizer.lr, self.clr())

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.trn_iterations)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

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
    parser.add_argument("--f_dim1", type=int, default=16)
    parser.add_argument("--f_dim2", type=int, default=1)
    parser.add_argument("--f_dim3", type=int, default=8)
    parser.add_argument("--nb_filters", type=int, default=512)
    parser.add_argument("--nb_epochs", type=int, default=100)
    parser.add_argument("--lr",type=float,default=.001)
    parser.add_argument("--loss",type=str,default='mse')
    parser.add_argument("--verbose",type=int,default=0)
    parser.add_argument("--batch_size",type=int,default=256)
    parser.add_argument("--dropout",type=float,default=0.0)
    parser.add_argument("--dataset",type=int,default=6)
    parser.add_argument("--use_previous_best",type=bool,default=False)
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
    parser.add_argument("--residualNet",type=bool,default=False)
    parser.add_argument("--depth",type=int,default=3)
    parser.add_argument("--AD",type=bool,default=False)
    args = parser.parse_args()
    return args

args = get_args()
print(args)


## Load model
if args.dataset==6:
    input_shape=(64,64,1)
elif args.dataset<11:
    input_shape=(196,196,1)
else:
    input_shape=(176,176,1)

if args.residualNet:
    model = cg.residual_projectionNet(depth=args.depth,k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters,input_shape=input_shape,dropout=args.dropout)
else:
    model = cg.linked_projection_network(num_stacks=0,regularizer_loss_weight=args.regularizer_loss_weight,input_shape=input_shape,relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters,elu=elu[i],dropout=args.dropout)

if args.num_gpus > 1: 
    model = make_parallel(model,args.num_gpus)
    args.batch_size = args.batch_size*args.num_gpus

model.name = 'deartifact_'+ '_' + str(args.f_dim1) + \
        '_' + str(args.f_dim2) + '_' + str(args.f_dim3)+ '_' + str(args.nb_filters)+ \
        '_lr' + str(args.min_lr)+ '_' + args.loss+  \
        '_numEpochs_' + str(args.nb_epochs) + '_DO'+ str(args.dropout) + \
        '_v' + str(args.dataset)+ \
        '_batchsize_' + str(args.batch_size) + \
        '_cycling_' + str(args.use_cycling_lr)+ '_minlr_' + str(args.min_lr) +  '_maxlr_' + str(args.max_lr) + '_lrss_' + str(args.lr_ss) +  \
        '_residualNet_'+ str(args.residualNet) + \
        '_depth_'+ str(args.depth) + \
        '_AD_' + str(args.AD) + \
        '_apr22'
print('model name: ' + model.name)

if args.use_previous_best:
    if args.best_is_old:
        best = hf.load_trained_CNN(name=model.name +'_old/best_simple',folder=output_folder)
    else:
        #best = hf.load_trained_CNN(name='deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_23_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked' + '/best_simple',folder='projection_results5/')
        #best = hf.load_trained_CNN(name='deartifact_False_0.300_16_1_8_325_False_0.0001_mse_0_100_0_False_0.0_False_False_19_False_0_False_500_0.5_1.0_num_stacks1_equalADFalse_num_normal_training_examples1000_march13_linked_old' + '/best_simple',folder='projection_results5/')
        # v25 
        #best = hf.load_trained_CNN(name='deartifact__16_1_8_325_0.001_mse_0_100_5_0.0_25_batchsize_64_cycling_False_minlr_0.001_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_march13_linked' + '/best_simple',folder='projection_results7/')
        # v19 AD
        best = hf.load_trained_CNN(name='deartifact__16_1_8_64_lr0.0001_mse_numEpochs_4000_DO0.0_v42_batchsize_64_cycling_False_minlr_0.0001_maxlr_0.005_lrss_5_residualNet_True_depth_20_AD_False_apr22' + '/2000_simple',folder='projection_results7/')
        #best = hf.load_trained_CNN(name='deartifact__16_1_8_325_0.001_mse_0_100_5_0.0_25_batchsize_64_cycling_False_minlr_0.001_maxlr_0.005_lrss_5_regLossWeight_0.0_longer_0_residualNet_True_depth_3_march13_linked' + '/best_simple',folder='projection_results7/')
        
        
    model.set_weights(best.get_weights())
    import time
    # rename files (which will be overwritten) to old/todays 
    fold = output_folder + model.name + '/'
    os.system('mv ' + fold + 'training_nums.out ' + fold + 'training_nums_' + time.strftime("%d_%m_%Y_%H_%M_%S") +'.out')
    os.system('mv ' + fold + 'training_plot.png ' + fold + 'training_plot_' + time.strftime("%d_%m_%Y_%H_%M_%S") +'.png')

opt = Adam(lr=args.min_lr)
#opt = SGD(lr=args.min_lr, momentum=0.9, decay=0.0001, nesterov=True)
loss_weights = []
loss_weights.append(1.0)

model.compile(loss=args.loss, optimizer=opt,loss_weights=loss_weights)


hf.generate_folder(output_folder+model.name)

clr = CyclicLR(base_lr=args.min_lr, max_lr=args.max_lr,
                    step_size=((10408./args.batch_size)*args.lr_ss)/50, 
                    mode='triangular2')

cb = generateImageCallback()

cb.set_training_nums_suffix('')
#     cb.set_inputs(X_test[0:8,:,:,:])
cb.set_name(model.name)


cb.set_num_epochs(0)
if args.residualNet:
    model_simple = cg.residual_projectionNet(depth=args.depth,k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters,input_shape=input_shape,dropout=args.dropout)
else:
    model_simple = cg.linked_projection_network(num_stacks=0,input_shape=input_shape,relu=args.relu,leaky_relu=lrelu[i],alpha=alpha[i],k1=args.f_dim1,k2=args.f_dim2,k3=args.f_dim3,nb_filters=args.nb_filters,elu=elu[i],longer=args.longer,dropout=args.dropout)

cb.set_simple_cnn(model_simple)

cbs = []
cbs.append(cb)
if args.use_cycling_lr:
    cbs.append(clr)

val_loss = hf.run_model_generator(model,dataset=args.dataset,nb_epoch=args.nb_epochs,batch_size=args.batch_size,callbacks=cbs,verbose=args.verbose,AD=args.AD,num_gpus=args.num_gpus)










# EoF #
