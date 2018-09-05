# Import standard modules
import os, pdb
import numpy as np

# Import modules for displaying results
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

# Import modules for deep learning
from utils_rice import *
from keras.callbacks import TensorBoard
from keras.optimizers import Adam


##################################################################################
######################      Create and train a model      ########################
##################################################################################

def createAndTrainModel(params):
                                        
    ############ Extract params ############
    learning_rate_base = params['learning_rate_base']
    kernel_size = params['kernel_size']
    num_epochs = params['num_epochs']
    batch_size = params['batch_size']
    dropout_rate = params['dropout_rate']
    activation_type = params['activation_type']
    growth_rate = params['growth_rate']
    num_nodes_fc = params['num_nodes_fc']
    USE_DATA_AUG = params['USE_DATA_AUG']
    num_layers_each_dense = params['num_layers_each_dense']
    compression_factor = params['compression_factor']
    rice_types = params['rice_types']
    normalization_type = params['normalization_type']
    N_classes = len(rice_types)
    
    
    ############ Load data ############
    print("--------------Load Data--------------")

    # Load training data and their corresponding labels
    x_training = np.load('x.npy')
    labels_training = np.load('labels.npy')
    
    # Normalize the data
    x_training = normalizeDataWholeSeed(x_training,normalization_type=normalization_type)
    
    # Extract some information
    num_training = x_training.shape[0]
    N_spatial = x_training.shape[1:3]
    N_bands = x_training.shape[3]
    num_batch_per_epoch = int(num_training/batch_size)
    
    print('#training = %d' %(num_training))
    print('#batches per epoch = %d' %(num_batch_per_epoch))
    
    print("--------------Done--------------")
    
    
    ############ Prepare the path for saving the models/stats ############
    print("--------------Prepare a path for saving the models/stats--------------")
    
    hparams = make_hyperparam_string(USE_DATA_AUG, learning_rate_base, batch_size, kernel_size, growth_rate, dropout_rate, num_training, num_nodes_fc, activation_type)
    print('Saving the model to...')
    
    results_dir = os.path.join(params['results_base_directory'],hparams)
    
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    print(results_dir)

    print("--------------Done--------------")

    ############ Create a model ############
    print("--------------Create a model--------------")
    
    # Generate a model
    model = DeepSpatioSpectral_CNN_classifier(data_num_rows=N_spatial[0], data_num_cols=N_spatial[1],num_classes=N_classes, kernel_size=kernel_size, growth_rate=growth_rate, num_layers_each_dense=num_layers_each_dense, compression_factor=compression_factor, activation_type=activation_type, dropout_rate=dropout_rate, num_input_chans=N_bands, num_nodes_fc=num_nodes_fc)   
    
    # Compile the model
    adam_opt = Adam(lr=learning_rate_base/batch_size, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=adam_opt)

    # Create a Tensorboard callback
    tbCallBack = TensorBoard(log_dir=results_dir, histogram_freq=0, write_graph=False, write_images=False)
    
    print("--------------Done--------------")

    ############ Train the model ############
    print("--------------Begin training the model--------------")

    # Possibly perform data augmentation
    from keras.preprocessing.image import ImageDataGenerator
    
    if USE_DATA_AUG:
        width_shift_range = 0.04
        height_shift_range = 0.04
        HORIZONTAL_FLIP = True
        VERTICAL_FLIP = True
        data_gen_args = dict(
            rotation_range=0.,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            horizontal_flip=HORIZONTAL_FLIP,
            vertical_flip=VERTICAL_FLIP,
            fill_mode = 'wrap')

        image_datagen = ImageDataGenerator(**data_gen_args)
    else:
        image_datagen = ImageDataGenerator()

    # Define a data generator to generate random batches
    def myGenerator(batch_size):
        for x_batch, y_batch in image_datagen.flow(x_training, labels_training, batch_size=batch_size, shuffle = True):
            yield (x_batch, y_batch)

    my_generator = myGenerator(batch_size)
    
    import time
    tic = time.clock()
    
    # Train the model
    hist = model.fit_generator(my_generator, steps_per_epoch=num_batch_per_epoch, epochs = num_epochs, initial_epoch = 0, verbose=2, callbacks = [tbCallBack])

    toc = time.clock()
    total_time = toc-tic
    print('Total training time = ' + str(total_time))
    
    print("--------------Done--------------")

    ############ Save the information ############
    print("--------------Save the information--------------")
    
    import pandas as pd
    
    # Save the trained model
    model.save(os.path.join(results_dir ,'final_model.h5'))
    
    # Extract the training loss   
    training_loss = hist.history['loss']

    # Save the training loss
    df = pd.DataFrame(data={'training loss': training_loss},index=np.arange(num_epochs)+1)
    df.to_csv(os.path.join(results_dir,'training_loss.csv'))
    
    # Save the training loss as a figure
    plt.figure(1)
    plt.title('Loss')
    plt.plot(training_loss, color='b',label='Training')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(results_dir,'training_loss.png'))
    plt.clf()   
    
    # Write a file with general information
    f = open(os.path.join(results_dir,'info.txt'),'w')
    f.write(hparams + '\n')
    f.write('Rice types = ' + str(rice_types)+'\n')
    f.write('Training time  = %f \n' %(total_time))    
    f.write('Normalization type = ' + str(normalization_type)+ '\n')
    f.write('# batches per epoch = %d \n' %(num_batch_per_epoch))
    f.write('# layers each dense = ' + str(num_layers_each_dense) + '\n')
    f.write('# epochs = ' + str(num_epochs) + '\n')
    f.write('# training  = %d \n' %(num_training))
    f.close()
    
    print("--------------Done--------------")
    
##################################################################################
###############################      Main Function     ###########################
##################################################################################      
if __name__ == '__main__':


    # Parameters (mostly determined using validation datasets)
    params = {}
    params['normalization_type'] = 'max'                      # Data normalization type 
    params['rice_types'] = ['DM','JP','RB','LP','KN','ML']    # Rice types
    params['activation_type'] = 'LeakyReLU'                   # Specify activation
    params['USE_DATA_AUG'] = True                             # Use data augmentation
    params['num_epochs'] = 800                                # Number of epochs
    params['batch_size'] = 3                                  # Batch size
    params['learning_rate_base'] = 0.0005                     # Initial learning rate
    params['kernel_size'] = 3                                 # Kernel size for a conv layer
    params['growth_rate'] = 20                                # Growth rate of a dense block
    params['num_layers_each_dense'] = [12,12,24,16]           # Number of layers in each dense block
    params['compression_factor'] = 0.6                        # Compression factor in a transition block
    params['dropout_rate'] = 0.05                             # Dropout rate
    params['num_nodes_fc'] = 512                              # Number of nodes in the fully-connected layer
    params['results_base_directory'] = './results/'
    
    # Add 'swish' activation
    if params['activation_type'] == 'swish':
        
        from keras.utils.generic_utils import get_custom_objects
        import keras.backend as K

        # Taken from https://github.com/dataplayer12/swish-activation/blob/master/MNIST/activations.ipynb
        def swish(x):
            beta=tf.Variable(initial_value=1.0,trainable=True)
            return x*tf.nn.sigmoid(beta*x) #trainable parameter beta

        get_custom_objects().update({'swish': swish})

    # Create and train a model
    createAndTrainModel(params)

