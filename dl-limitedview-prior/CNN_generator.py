# CNN generator for 'grid' searching for some task
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten,Input,merge
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D,GlobalAveragePooling2D, Lambda, Merge,Cropping2D,GlobalMaxPooling2D,RepeatVector,Reshape
from keras.layers.advanced_activations import LeakyReLU, ELU
from keras.layers import merge
import math
# from keras.layers.containers import Graph
from keras import regularizers


def simple_CNN1(nb_dense=128,final_relu=True,global_pool=True,nb_blocks=5,nb_layers=1,nb_filters=32,pool_size=(2,2),kernel_size=(3,3),dropout=.5,input_shape=(256,256,1)):
    
    # Generic Conv module, no fanciness
    def add_module(model,kernel_size=kernel_size):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        model.add(Activation('relu'))
        if dropout >0:
            model.add(Dropout(dropout))
    
    model = Sequential()
    model.add(Convolution2D(nb_filters,kernel_size[0],kernel_size[0],input_shape=input_shape))
    
    # Conv layers, nb_blocks and nb_layers
    for i in range(nb_blocks):
        for j in range(nb_layers):
            add_module(model)
        
        model.add(MaxPooling2D(pool_size=pool_size))

    # Conv then pool before final layer
    add_module(model)
    if global_pool:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
    
    # Final Dense layer prior to output, possible relu layer as well
    model.add(Dense(nb_dense))
    if final_relu:
        model.add(Activation('relu'))
    
    # Final output layer, will need some kind of optimizer on this before fitting
    model.add(Dense(1, init='normal'))

    return model

def simple_CNN2(nb_dense=128,final_relu=False,global_pool=True,nb_blocks=3,nb_layers=2,nb_filters=32,pool_size=(2,2),kernel_size=(3,3),dropout=.5,input_shape=(256,256,1),matmul=False):
    
    
    def add_module(model,dropout=.5,kernel_size=(3,3)):
        model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
        if dropout >0:model.add(Dropout(dropout))
        model.add(Activation('relu'))
        

    model = Sequential()
    # Layer 1
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1],
                            border_mode='valid',
                            input_shape=input_shape))
    if dropout >0:model.add(Dropout(dropout))
    model.add(Activation('relu'))

    for j in range(nb_blocks):
        if j==0:
            for i in range(nb_layers-1):
                add_module(model,dropout)
        else:
            for i in range(nb_layers):
                add_module(model,dropout)

        model.add(MaxPooling2D(pool_size=pool_size))

    
    model.add(Convolution2D(nb_filters, kernel_size[0], kernel_size[1]))
    model.add(Dropout(dropout))
    model.add(Activation('relu'))
    
    if global_pool:
        model.add(GlobalAveragePooling2D())
    else:
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Flatten())
    
    model.add(Dense(nb_dense))

    model.add(Dense(1, init='normal'))
    if matmul:
        model.add(Lambda(lambda x: x ** 2))
    
    return model


def densenet_1(nb_blocks=3,nb_layers=2,dropout=.5,input_shape=(256,256,1),growth_rate=12,init_channels=16):
    import densenet
    model = densenet.densenet_model_regression(input_shape,nb_blocks, nb_layers, growth_rate, dropout=dropout, l2_reg=1e-4,
                   init_channels=init_channels)
    return model


def projection_network(relu=False,leaky_relu=False,nb_filters=512,k1=16,k2=1,k3=8,input_shape=(64,64,1),alpha=0.3,elu=False,longer=0,dropout=.3,final_act=True):
    
    def add_activation(model):
        if leaky_relu:
            model.add(LeakyReLU(alpha=alpha))
        elif elu:
            model.add(Activation('elu'))
        elif relu:
            model.add(Activation('relu'))
        else:
            model.add(Activation('tanh'))
    
    if leaky_relu and elu:
        print('For some reason, we have leaky relu and elu set to true... setting elu to true')
        elu= True
        leaky_relu=False
    
    model = Sequential()
    model.add(Convolution2D(nb_filters, k1, k1, input_shape=input_shape, border_mode='valid'))
    add_activation(model)
    if dropout>0: model.add(Dropout(dropout))
    
    for i in range(longer):
        model.add(Convolution2D(nb_filters, k2, k2))
        add_activation(model)
        if dropout>0: model.add(Dropout(dropout))
    
    model.add(Convolution2D(nb_filters, k2, k2))
    add_activation(model)
    if dropout>0: model.add(Dropout(dropout))
    
    
    model.add(Convolution2D(input_shape[2], k3, k3))
    if final_act: model.add(Activation('tanh'))
    
    return model
    
# mar 8
# 
def g_tv_to_ssim_network(input_shape=(100,256,1),nb_filters=32,k1=5,pool_size=(2,2)):

    input_img= Input(shape=input_shape,name='input_img')
    input_tv = Input(shape=(1,1,1),name='input_tv')
    input_tv_flat = Flatten()(input_tv)
    
    c1 = Convolution2D(nb_filters, k1, k1, activation='relu', border_mode='valid')(input_img) # then call the layer
    
    max_pool = MaxPooling2D(pool_size=pool_size)
    mp1 = max_pool(c1)
    
    c2 = Convolution2D(nb_filters*2, k1, k1, activation='relu', border_mode='valid')(mp1) # then call the layer
    
    mp2 = max_pool(c2)
    
    flat_fc = Flatten()(mp2)
    x = merge([flat_fc, input_tv_flat], mode='concat')
    
    x = Dense(128, activation='relu')(x)
    x = merge([x, input_tv_flat], mode='concat')
    output = Dense(1)(x)

    return Model(input=[input_img, input_tv], output=output)     
    
# mar 13
def linked_projection_network(num_stacks=1,regularizer_loss_weight=0.,relu=False,leaky_relu=False,nb_filters=512,k1=16,k2=1,k3=8,input_shape=(64,64,1),alpha=0.3,elu=False,longer=0,dropout=.3,final_act=True):
    input_img = Input(shape=input_shape)

    conv1 = Convolution2D(nb_filters, k1, k1, activation='tanh', border_mode='valid') # create the layer instance that i want to tie to
    c1 = conv1(input_img) # then call the layer

    conv2 = Convolution2D(nb_filters, k2, k2, activation='tanh', border_mode='valid') # create the layer instance that i want to tie to
    c2 = conv2(c1)

    conv3 = Convolution2D(1, k3, k3, activation='tanh', border_mode='valid',name='aux_output') # create the layer instance that i want to tie to
    c3 = conv3(c2)
    
    if num_stacks==0:
        model = Model(input=[input_img], output=[c3])
        return model
    
    pad = int((k1+k2+k3-3)/2)
    zpad = ZeroPadding2D(padding=(pad, pad), dim_ordering='default')
    z1 = zpad(c3)

    c4 = conv1(z1)

    c5 = conv2(c4)

    c6 = conv3(c5)

    if num_stacks==1:
        model = Model(input=[input_img], output=[c6, c3]) # Goes [main output, auxiliary output(s)]
        return model
    
    z2 = zpad(c6)

    c7 = conv1(z2)

    c8 = conv2(c7)

    c9 = conv3(c8)
    
    if num_stacks==2:
        model = Model(input=[input_img], output=[c9,c6, c3,]) # Goes [main output, auxiliary output(s)]
        return model
    
    z3 = zpad(c9)

    c10 = conv1(z3)

    c11 = conv2(c10)

    c12 = conv3(c11)

    if num_stacks==3:
        model = Model(input=[input_img], output=[c12,c9,c6, c3,]) # Goes [main output, auxiliary output(s)]
        return model

def residual_projectionNet(depth=3,nb_filters=512,input_shape=(64,64,1),dropout=0):
    
    
    # Lets make a really deep CNN 
    input_img = Input(shape=input_shape)
    
    model = Convolution2D(nb_filters,3,3,activation='relu', border_mode='valid')(input_img)
    if dropout>0: model = Dropout(dropout)(model)
    
    for i in range(depth-2):
        model = Convolution2D(nb_filters,3,3,activation='relu', border_mode='valid')(model)
        if dropout>0: model = Dropout(dropout)(model)


    model = Convolution2D(1,3,3, border_mode='valid')(model)
    
    crop_amount = int(int(input_img.get_shape()[1] - model.get_shape()[1])/2)
    
    crop = Cropping2D(cropping=((crop_amount,crop_amount),(crop_amount,crop_amount)))(input_img)
    
    merge1 = merge([crop, model],mode='sum')
    
    final_model = Model(input=[input_img], output=[merge1])
    return final_model
    
def residual_projectionNet2(depth=3,nb_filters=512,k1=16,k2=1,k3=8,input_shape=(64,64,1),dropout=0):
    
    
    # Lets make a really deep CNN 
    input_img = Input(shape=input_shape)
    
    model = Convolution2D(nb_filters,3,3,activation='relu', border_mode='same')(input_img)
    if dropout>0: model = Dropout(dropout)(model)
    
    for i in range(depth-2):
        model = Convolution2D(nb_filters,3,3,activation='relu', border_mode='same')(model)
        if dropout>0: model = Dropout(dropout)(model)


    model = Convolution2D(1,3,3, border_mode='same')(model)
    
    crop_amount = int(int(input_img.get_shape()[1] - model.get_shape()[1])/2)
    
    crop = Cropping2D(cropping=((crop_amount,crop_amount),(crop_amount,crop_amount)))(input_img)
    
    merge1 = merge([crop, model],mode='sum')
    
    final_model = Model(input=[input_img], output=[merge1])
    return final_model

# 
# def VNet(depth=3,nb_filters=32,input_shape=(256,256,1),dropout=0,sz_filters=3,theta=80,nrays=256):
# 	
# 	input_img = Input(shape=input_shape)
# 	input_img_y = Input(shape=input_shape)
# 	H_shape = (256,256)
# 	H = Input(shape=H_shape)
# 	
# 	G = K.tf.matmul(H,K.tf.reshape(input_img_y,(1,256*256)))
# 	
# 	
# 	def add_deep_stuff(x):
# 		x = Convolution2D(nb_filters,sz_filters,sz_filters, border_mode='same')(x)
# 		x = Convolution2D(nb_filters,sz_filters,sz_filters, border_mode='same')(x)
# 		x = Convolution2D(nb_filters,sz_filters,sz_filters, border_mode='same')(x)
# 		x = Convolution2D(input_shape(2),1,1, border_mode='same')(x)
# 		return x
# 	
# 	def add_scale_layer(x):
# 		# Getting a tensor with a single 1:
# 		single_one = GlobalMaxPooling2D()(x)
# 		single_one = Lambda(lambda x: x**0)(single_one)
# 		# Scaling parameter!
# 		trainable_scaling_parameter = Dense(1)(single_one)
# 		# Repmat this to the size of conv1
# 		size_conv1_output = x.get_shape()[1]*x.get_shape()[2]*x.get_shape()[3]
# 		repeated_scale = RepeatVector(int(size_conv1_output))(trainable_scaling_parameter)
# 		target_shape = (int(x.get_shape()[1]),int(x.get_shape()[2]),int(x.get_shape()[3]))
# 		reshaped_to_conv1_shape = Reshape(target_shape)(repeated_scale)
# 		
# 		# Finally multiply the scaling matrix with our convolution output:
# 		scaled_conv1_output = merge([x,reshaped_to_conv1_shape],mode='mul')
# 		return scaled_conv1_output
# 	
# 	def add_gradient_layer(x):
# 		inner = K.tf.matmul(H,K.tf.reshape(x,(1,256*256)))
# 		dif = K.tf.subtract(inner,G)
# 		gradient = K.tf.matmul(H,dif,adjoint_a=True)
# 		gradient = add_scale_layer(gradient)
# 		return gradient
# 		
# 		
# 	
# 	model = input_img
# 	# Do a couple blocks
# 	for i in range(depth):
# 		model = add_deep_stuff(model)
# 		gradient = add_gradient_layer(model)
# 		model = add_scale_layer(model)
# 		model = merge([model, gradient],mode='sum')
# 	
# 	final_model = Model(input=[input_img,input_img,y,H])
# 	return final_mode
		
		

	


# Hacky stuff to implement scaling layer
# input_shape=(64,64,1)
# input_img = Input(shape=input_shape)
# conv1_output = Convolution2D(64,3,3,activation='relu', border_mode='valid')(input_img)
# 
# # Getting a tensor with a single 1:
# single_one = GlobalMaxPooling2D()(input_img)
# single_one = Lambda(lambda x: x**0)(single_one)
# # Scaling parameter!
# trainable_scaling_parameter = Dense(1)(single_one)
# # Repmat this to the size of conv1
# size_conv1_output = conv1_output.get_shape()[1]*conv1_output.get_shape()[2]*conv1_output.get_shape()[3]
# repeated_scale = RepeatVector(int(size_conv1_output))(trainable_scaling_parameter)
# target_shape = (int(conv1_output.get_shape()[1]),int(conv1_output.get_shape()[2]),int(conv1_output.get_shape()[3]))
# reshaped_to_conv1_shape = Reshape(target_shape)(repeated_scale)
# 
# # Finally multiply the scaling matrix with our convolution output:
# scaled_conv1_output = merge([conv1_output,reshaped_to_conv1_shape],mode='mul')












# EoF #