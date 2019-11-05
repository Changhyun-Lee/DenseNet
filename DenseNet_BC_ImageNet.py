# -*- coding: utf-8 -*-
"""
@author: changhyun1.lee@gmail.com
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import ReLU
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import BatchNormalization


def conv_block(x, growth_rate, dropout_rate=None):
        
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=growth_rate*4, kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False)(x)
    if dropout_rate : x = Dropout(dropout_rate)(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=growth_rate, kernel_size=(3,3), strides=(1,1), padding='same', use_bias=False)(x)
    if dropout_rate : x = Dropout(dropout_rate)(x)
    
    return x
    

def dense_block(x, nb_layers, nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True):
    
    concat_feat = x 
        
    for i in range(nb_layers):
        x = conv_block(concat_feat, growth_rate, dropout_rate)
        concat_feat = concatenate([concat_feat, x], axis=-1)

        if grow_nb_filters:
            nb_filter += growth_rate

    return concat_feat, nb_filter    


def transition_block(x, nb_filter, compression=1.0, dropout_rate=None):

    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Conv2D(filters=int(nb_filter*compression), kernel_size=(1,1), strides=(1,1), padding='same', use_bias=False)(x)
    if dropout_rate : x = Dropout(dropout_rate)(x)

    x = AveragePooling2D((2, 2), strides=(2, 2))(x)

    return x, int(nb_filter*compression)    


def densenet(input_shape, nb_layers, nb_dense_block, growth_rate,
             nb_filter, dropout_rate, weight_decay, classes, theta):
    # imageNet size
    inputs = Input(shape=(224, 224, 3), name='data')
    print('Input : ', inputs.shape)
    
    # Convolution layer
    x = Conv2D(filters=nb_filter, kernel_size=(7,7), strides=(2,2), padding='same', use_bias=False)(inputs)
    print('Conv2D : ', x.shape) 
    
    # pooling layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    print('MaxPooling2D : ', x.shape)
    
    # Dense block
    for block_idx in range(nb_dense_block-1):
        x, nb_filter = dense_block(x, nb_layers[block_idx], nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True)
        print('dense_block %d: '%(block_idx+1), x.shape)
        x, nb_filter = transition_block(x, nb_filter, compression=theta, dropout_rate=None)
        print('transition_block %d: '%(block_idx+1), x.shape)
    
    x, nb_filter = dense_block(x, nb_layers[block_idx+1], nb_filter, growth_rate, dropout_rate=None, grow_nb_filters=True)
    print('dense_block %d: '%(block_idx+2), x.shape)
    
    # Classification layer
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    print('GlobalAveragePooling2D : ', x.shape)
    
    x = Dense(classes)(x)
    x = Activation('softmax')(x)
    
    model = Model(inputs, x, name='densenet')
    
    return model


# Model generation
# k=32, L=121, For DenseNet-121
input_shape = (224, 224, 3)
nb_layers = [6,12,24,16] 
nb_dense_block=len(nb_layers)
growth_rate=32
nb_filter=growth_rate*2
dropout_rate=0.0
weight_decay=1e-4
classes=10
theta=0.5

model = densenet(input_shape=input_shape, 
                 nb_layers=nb_layers,
                 nb_dense_block=nb_dense_block,
                 growth_rate=growth_rate,
                 nb_filter=nb_filter,
                 dropout_rate=dropout_rate,
                 weight_decay=weight_decay,
                 classes=classes,
                 theta=theta)

model.summary()



