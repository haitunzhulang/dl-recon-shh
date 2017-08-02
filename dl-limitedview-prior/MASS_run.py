
import CNN_generator as cg
import helper_functions as hf
from keras import backend as K
import glob

data_version =2


attempt = 12.4
loss='mean_squared_error'
matmul = False

if attempt ==4:
    nb_dense = [64]
    final_relu = [True]
    global_pool = [False]
    nb_blocks = [2,3,5]
    nb_layers = [1,2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    

if attempt ==5:
    nb_dense = [64]
    final_relu = [True]
    global_pool = [False, True]
    nb_blocks = [2,3,5]
    nb_layers = [1]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (5,5)
    dropout=.5

if attempt==6:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [18]
    
if attempt==7:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [1,2,4]
    nb_layers = [1,2,3]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
   

if attempt==8:
    nb_dense = [64,128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [2,3]
    nb_layers = [2,3]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (5,5)
    dropout=.5

if attempt==9:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [13,18,25,30,50]
    
if attempt==10:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [25]
    DA = [False]
    
if attempt==11:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [25]
    DA = [True]
    
if attempt==12:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [25]
    DA = [True]
    data_version=5
    
if attempt==13:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3]
    nb_layers = [2]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=0
    nb_epoch = [50]
    growth_rate = 12
    init_channels=16
    
    data_version=5
    DA = [True]

if attempt==12.1:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [4]
    nb_layers = [3,4,5]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [50]
    DA = [True]
    data_version=5

if attempt==12.2:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3,4]
    nb_layers = [3,4]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=.5
    nb_epoch = [100]
    DA = [True]
    data_version=2
    loss = 'mae'

if attempt==12.3:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3,4]
    nb_layers = [3,4]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=0
    nb_epoch = [100]
    DA = [True]
    data_version=2
    loss = 'mae'
    
if attempt==12.4:
    nb_dense = [128]
    final_relu = [False]
    global_pool = [True]
    nb_blocks = [3,4]
    nb_layers = [3,4]
    nb_filters=[32]
    pool_size = (2,2)
    kernel_size = (3,3)
    dropout=0
    nb_epoch = [100]
    DA = [True]
    data_version=2
    loss = 'mae'
    matmul=True
    

X_train,X_test,Y_train,Y_test = hf.load_data(version=data_version)

print('**********************************************************')
print ('Starting attempt ' + str(attempt) + '!')
print('**********************************************************')

import time
t = time.time()
drops=0
batch_size=64
display=False

for i in range(len(nb_dense)):
    for j in range(len(final_relu)):
        for k in range(len(global_pool)):
            for l in range(len(nb_blocks)):
                for m in range(len(nb_layers)):
                    for n in range(len(nb_filters)):
                        for o in range(len(nb_epoch)):
                            for p in range(len(DA)):
                                print('i:' + str(i) +',j:' +str(j) + ',k:'+str(k)+',l:'+str(l)+',m:'+str(m)+',n:'+str(n)+',o:'+str(o)+',p:'+str(p))
                                K.clear_session()
                                if attempt<5:
                                    model = cg.simple_CNN1(nb_dense=nb_dense[i],final_relu=final_relu[j],global_pool=global_pool[k],nb_blocks=nb_blocks[l],nb_layers=nb_layers[m],nb_filters=nb_filters[n],pool_size=pool_size,kernel_size=kernel_size,dropout=dropout,input_shape=(256,256,1))
                                elif attempt < 13:
                                    model = cg.simple_CNN2(nb_dense=nb_dense[i],final_relu=final_relu[j],global_pool=global_pool[k],nb_blocks=nb_blocks[l],nb_layers=nb_layers[m],nb_filters=nb_filters[n],pool_size=pool_size,kernel_size=kernel_size,dropout=dropout,input_shape=(256,256,1),matmul=matmul)
                                else:
                                    model = cg.densenet_1(nb_blocks=nb_blocks[i],nb_layers=nb_layers[m],dropout=dropout,input_shape=(256,256,1),growth_rate=growth_rate,init_channels=init_channels)
                                
                                name = 'Attempt'+str(attempt)+'_' + str(model.count_params())+'_Dataset_v' + str(data_version) + '_' + str(nb_dense[i]) + '_' + str(final_relu[j]) + '_' + str(global_pool[k]) + '_'
                                name = name + str(nb_blocks[l]) + '_' + str(nb_layers[m]) + '_' + str(nb_filters[n]) + '_' + str(nb_epoch[o]) + '_' + str(DA[p])
                                print('Starting : ' + name)
                                
                                model.name = name
                                model.compile(loss=loss, optimizer='adadelta')
                                val_loss = hf.run_model(model,X_train,Y_train,X_test,Y_test,nb_epoch=nb_epoch[o],drops=drops,batch_size=batch_size,DA=DA[p])
                                hf.save_model(model,name=name)
                                outputs = hf.predict_outputs(model,X_test,target_dim = (256,256,1))
                                hf.display_distribution_of_outputs_regression(outputs,Y_test,name,display=display,output_folder=name)
                                if attempt <5:
                                    model2 = cg.simple_CNN1(nb_dense=nb_dense[i],final_relu=final_relu[j],global_pool=global_pool[k],nb_blocks=nb_blocks[l],nb_layers=nb_layers[m],nb_filters=nb_filters[n],pool_size=pool_size,kernel_size=kernel_size,dropout=0,input_shape=(256,256,1))
                                elif attempt < 13:
                                    model2 = cg.simple_CNN2(nb_dense=nb_dense[i],final_relu=final_relu[j],global_pool=global_pool[k],nb_blocks=nb_blocks[l],nb_layers=nb_layers[m],nb_filters=nb_filters[n],pool_size=pool_size,kernel_size=kernel_size,dropout=0,input_shape=(256,256,1))
                                else:
                                    model2 = cg.densenet_1(nb_blocks=nb_blocks[i],nb_layers=nb_layers[m],dropout=0,input_shape=(256,256,1),growth_rate=growth_rate,init_channels=init_channels)
                                
                                hf.save_model(model2,name=name +'_deploy',weights=False)
                                #hf.PLS(name,display=display,output_folder=name)
                                print(name + ' took ' + str(time.time()-t) + ' s')
                                t = time.time()
                                intermediate = glob.glob("PLS_Out/" + name + "/weights-improvement*")
                                intermediate.sort()
                                for i in range(len(intermediate)):
                                    hf.PLS(name,display=False,output_folder=name,weight_file=intermediate[i],short=True,short_index=str(i))


print('**********************************************************')
print ('Done with attempt ' + str(attempt) + '!')
print('**********************************************************')


#hf.PLS(name,index=8018,display=False,output_folder=name)





# EoF #