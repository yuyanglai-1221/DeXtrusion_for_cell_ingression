import os, random, math
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPool2D, GlobalMaxPool2D, Input
from tensorflow.keras.layers import TimeDistributed, GRU, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.models import save_model, load_model
from tensorflow.keras.callbacks import Callback, ReduceLROnPlateau, ModelCheckpoint 
from tensorflow.keras.optimizers import SGD
from sklearn import metrics
from IPython.display import clear_output
import tensorflow as tf
from tensorflow.keras.backend import clear_session
from tensorflow.keras.backend import function

"""
BSD 3-Clause License

Copyright (c) 2022, Gaëlle  LETORT and Alexis Villars
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""

class Network:
    """ DeXNet: dextrusion neural network
    
    Allow to configure, train or retrain DeXNets. This class is called from DeXtrusion main functions.
    """
    def __init__(self, verbose=True):
        self.init_tf(verbose)
        self.model = None

    def create_model(self, shape=(10,45,45), ncat=4, nb_filters=8):
        self.model = self.action_model(shape, ncat, nb_filters)
        self.model.compile( SGD(0.1), 'categorical_crossentropy', metrics=['acc'] )

    def save(self, model_path):
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        save_model(self.model, model_path)
        
    def init_tf(self, verbose=True):
        if verbose:
            print("Tensorflow with Cuda: "+str(tf.test.is_built_with_cuda()))
            print("Tensorflow version: "+str(tf.__version__))
            print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
        config = tf.compat.v1.ConfigProto( device_count = {'GPU': 0} )
        sess = tf.compat.v1.Session(config=config)
        tf.compat.v1.keras.backend.set_session(sess)
        
    def reset(self, model_path):
        clear_session()
        self.model = load_model(model_path)

    

    #### Neural network architecture functions
    
    def get_timedis_layer(self):
         input_tensor = Input(self.shape[1:]+(1,), name='input')
         #layer = Model(inputs=self.model.input,
         #                              outputs=self.model.get_layer("dense_1").output)
         layer = self.model.get_layer("time_distributed").output
         return Model(self.model.input, layer)
    
    def get_last_layer(self, x):
        layer = self.model.get_layer("dense").output
        get_layer_output = function(self.model.layer[0].input, layer)
        return get_layer_output([x])
    
    def conv_block(self,input_tensor, nfil, momentum):
        x = Conv2D(nfil, (3,3), padding='same', activation='relu')(input_tensor)
        x = Conv2D(nfil, (3,3), padding='same', activation='relu')(x)
        x = BatchNormalization(momentum=momentum)(x)
        return x
        
    def build_convnet(self,input_tensor, nfilters):
        momentum = .95
        c1 = self.conv_block(input_tensor, nfilters, momentum)
        c1 = MaxPool2D()(c1)
        c2 = self.conv_block(c1, nfilters*2, momentum)
        c2 = MaxPool2D()(c2)
        c3 = self.conv_block(c2, nfilters*4, momentum)
        c3 = MaxPool2D()(c3)
        c4 = self.conv_block(c3, nfilters*8, momentum)
        c4 = GlobalMaxPool2D()(c4)
        return Model(inputs=[input_tensor], outputs=[c4])

    def action_model(self, shape, ncat, nfilters):
        input_tensor = Input(shape[1:]+(1,), name='input')
        convnet = self.build_convnet(input_tensor, nfilters)
        
        inputl = Input(shape+(1,), name='input')
        tconv = TimeDistributed(convnet)(inputl) 
        de = GRU(64)(tconv)   # 64 GRU or LSTM
        #tconv = LSTM(24)(tconv)   # GRU or LSTM
        
        # Decision network
        de = Dropout(.5)(de)
        de = Dense(32, activation='relu')(de)
        de = Dense(16, activation='relu')(de)
        #de = Dropout(.5)(de)
        #de = Dense(8, activation="relu")(tconv)
        output = Dense(ncat, activation='softmax')(de)
        return Model(inputs=[inputl], outputs=[output])
    
    #### Neural network training functions

    def train(self, train_gen, validation_gen, epochs, plot=True):
        if plot:
            plot_losses = TrainingPlot()
            callbacks = [ ReduceLROnPlateau(verbose=1), plot_losses ]
            self.model.fit(train_gen, validation_data=validation_gen, verbose=1, epochs=epochs, callbacks=callbacks)
        else:
            self.model.fit(train_gen, validation_data=validation_gen, verbose=1, epochs=epochs)
    

    #### Prediction/Evaluation functions

    def evaluate_prediction(self, datatest):
        ''' Evaluate prediction score function '''
        for i in range(0, len(datatest)):
            temp_var = datatest.__getitem__(i)
            a = temp_var[0]
            b = self.model.predict(a)
            if i == 0:
                y_pred = np.argmax(b, axis = 1)
                y_true = np.argmax(temp_var[1], axis = 1)
            else:
                y_pred = np.append(y_pred,np.argmax(b, axis = 1))
                y_true = np.append(y_true, np.argmax(temp_var[1], axis = 1))
    
        print("---------------------------")
        print("-- Scores")
        print("Nb test data: "+str(y_true.shape))  
        print("Accuracy: "+str(metrics.accuracy_score(y_true, y_pred)))
        print("Balanced accuracy: "+str(metrics.balanced_accuracy_score(y_true, y_pred)))
        print("Confusion matrix:")
        print(metrics.confusion_matrix(y_true, y_pred))

    def evaluate_batch(self, datatest, i, what='false_death'):
        temp_var = datatest.__getitem__(i)
        a = temp_var[0]
        b = self.model.predict(a)
        y_pred = np.argmax(b, axis = 1)
        y_true = np.argmax(temp_var[1], axis = 1)
        
        if what == 'false_death':
            return (np.intersect1d(np.where(y_true!=1)[0], np.where(y_pred==1)[0]))
        if what == 'false_sop':
            return (np.intersect1d(np.where(y_true!=2)[0], np.where(y_pred==2)[0]))
        if what == 'nothing_death':
            return (np.intersect1d(np.where(y_true==0)[0], np.where(y_pred==1)[0]))
        return None
        
    def predict_batch(self, img_batch):
        return self.model.predict(img_batch)
        
    def predict_convolution(self, datatest):
        ''' Evaluate prediction score function '''
        td = self.get_timedis_layer()
        for i in range(0, len(datatest)):
            temp_var = datatest.__getitem__(i)
            a = temp_var[0]
            b = td.predict(a)
            j = 0
            for im in b:
                datatest.write_img(im, i, j)
                j = j + 1

    

class TrainingPlot(Callback):
    ''' Handle the visualization of the training loss and accuracy at each epochs ''' 

    def on_train_begin(self, logs={}):
        ''' Initialize the logs when training begins'''
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []
    
    def on_epoch_end(self, epoch, logs={}):
        ''' Called at the end of each epoch plot the loss and accuracy '''
        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))
        
        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:
            # Clear the previous plot
            clear_output(wait=True)
            N = np.arange(0, len(self.losses))
            # print(plt.style.available) to see the available options
            plt.style.use("seaborn")
            
            # Plot train loss, train acc, val loss and val acc against epochs passed
            fig, axs = plt.subplots(1, 2)
            axs[0].plot(N, self.losses, label = "train_loss")
            axs[0].plot(N, self.val_losses, label = "val_loss")
            axs[1].plot(N, self.acc, label = "train_acc")
            axs[1].plot(N, self.val_acc, label = "val_acc")
            
            axs[0].set_title("Loss SGD [Epoch {}]".format(epoch))
            axs[0].set(xlabel="Epoch #", ylabel="loss")
            axs[1].set_title("accuracy [Epoch {}]".format(epoch))
            axs[1].set(xlabel="Epoch #", ylabel="accuracy")
            axs[1].set_ylim([0.3,1])
            plt.legend()
            plt.show()

