import numpy as np
from matplotlib import pyplot as plt
import math, sys

import keras
import keras.backend as K
import tensorflow as tf
from keras.layers import Input, Conv2D, MaxPooling2D, Activation, BatchNormalization, Add, Conv2DTranspose, Flatten, Dense, Conv1D, AveragePooling2D, LeakyReLU, PReLU
from keras.layers.core import Dropout
from keras.layers.merge import concatenate
from keras.models import Model

##################################################################################
#######################      Data Normalization     ##############################
##################################################################################

## Normalize each datacubes
def normalizeDataWholeSeed(data,normalization_type='max'):
    
    if normalization_type == 'max':
        for idx in range(data.shape[0]):
            data[idx,:,:,:] = data[idx,:,:,:]/np.max(abs(data[idx,:,:,:]))
            
    elif normalization_type == 'l2norm':
        from numpy import linalg as LA
        for idx in range(data.shape[0]):
            data[idx,:,:,:] = data[idx,:,:,:]/LA.norm(data[idx,:,:,:])        
        
    return data

##################################################################################
######################      Hyperparameter String     ############################
##################################################################################

# Make the hyperparam list
def make_hyperparam_string(USE_DATA_AUG, learning_rate_base, batch_size, kernel_size, growth_rate, dropout_rate, num_training, num_nodes_fc, activation_type, INCLUDE_DATE = False):
    
    # Date and time
    if INCLUDE_DATE:
        import datetime
        now = datetime.datetime.now()
        hparam = str(now.year)
        if now.month < 10:
            hparam += "0" + str(now.month)
        else:
            hparam += str(now.month)

        if now.day < 10:
            hparam += "0" + str(now.day) + "_"
        else:
            hparam += str(now.day) + "_"
    else:
        hparam = ""
    
    # Hyper-parameters
    if USE_DATA_AUG:
        hparam += "AUG_"

    hparam += str(num_nodes_fc) + "nodes_"+ str(learning_rate_base) + "lr_" + str(batch_size) + "batch_"+ str(kernel_size) + "kernel_" + str(growth_rate) + "growth_" + str(dropout_rate) + "drop_"+ str(num_training) + "train_" + activation_type
    
    return hparam


##################################################################################
##########################      Confusion Matrix     #############################
##################################################################################

# Print and plot a confusion matrix
# Taken from http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):

    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.clim(0,sum(cm[0,:]))
    plt.xlabel('Predicted label')
    
##################################################################################
#######################      Spatio-spectral deep CNN     ########################
##################################################################################

# Batch norm -> Activation -> 1x1 conv2D (bottleneck) -> 3x3 conv2D -> dropout
# growth_rate is 'k' in the original DenseNet paper
def conv2D_DeepSpatioSpectral_CNN(x, growth_rate, kernel_size, activation_type, dropout_rate):
    
    # Batch norm
    x = BatchNormalization()(x)
    
    # Activation
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)
    else:
        x = Activation(activation_type)(x)
    
    # 1x1 Conv2D: Here, we produce 4 * growth_rate as in the paper (4k filters)
    x = Conv2D(4*growth_rate, kernel_size=1, activation=None, use_bias=False, padding='same', kernel_initializer='truncated_normal')(x)
    
    # 3x3 Conv2D
    x = Conv2D(growth_rate, kernel_size, activation=None, use_bias=True, padding='same', kernel_initializer='truncated_normal')(x)
    
    # Dropout
    return Dropout(dropout_rate)(x)

    
def createDenseBlock(x, num_layers, growth_rate, kernel_size, activation_type, dropout_rate):
    
    # Store x0
    x_memory = x
    
    for idx_layer in range(num_layers):
        
        # Compute x_l
        x = conv2D_DeepSpatioSpectral_CNN(x_memory, growth_rate, kernel_size, activation_type, dropout_rate)
        
        # Store [x0,x1,...,x_l]
        x_memory = concatenate([x_memory, x], axis=3)
        
    return x_memory

def createTransitionLayer(x, compression_factor, num_input_filters, growth_rate, kernel_size, activation_type, dropout_rate):

    x = BatchNormalization()(x)
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)        
    else:
        x = Activation(activation_type)(x)
          
    x = Conv2D(int(compression_factor*num_input_filters),kernel_size=1, use_bias=False)(x)
    
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return AveragePooling2D((2, 2), strides=(2, 2))(x)


# growth_rate: number of filters for each normal convolution ('k' in the paper)
def DeepSpatioSpectral_CNN_classifier(data_num_rows, data_num_cols, num_classes, kernel_size=3, growth_rate=12, num_layers_each_dense=[6,12,24,16], compression_factor = 0.5, activation_type='swish', dropout_rate=0.05, num_input_chans=1, num_nodes_fc=64):
    
    input_data = Input(shape=(data_num_rows, data_num_cols, num_input_chans))  # change this if using `channels_first` image data format
    
    # Input layer: Conv2D -> batch norm -> activation
    x = Conv2D(num_input_chans, kernel_size+6, activation=None, use_bias=True, padding='same', kernel_initializer='truncated_normal')(input_data)
    
    # Dense blocks & Transition Layers
    for idx_dense_block in range(len(num_layers_each_dense)):

        x = createDenseBlock(x, num_layers_each_dense[idx_dense_block], growth_rate, kernel_size, activation_type, dropout_rate)
        
        num_input_filters = int(x.shape[3])
        x = createTransitionLayer(x, compression_factor, num_input_filters, growth_rate, kernel_size, activation_type, dropout_rate)
        
    # Output layer
    x = Flatten()(x)
    x = Dense(units = num_nodes_fc, activation=None, kernel_initializer='truncated_normal')(x)
    
    if activation_type == 'LeakyReLU':
        x = LeakyReLU()(x)
    elif activation_type == 'PReLU':
        x = PReLU()(x)        
    else:
        x = Activation(activation_type)(x)
        
    output_data = Dense(units = num_classes, activation='softmax', kernel_initializer='truncated_normal')(x)
    
    return Model(inputs=input_data, outputs=output_data)
