from numpy.random import seed
seed(1)
import tensorflow as tf
tf.random.set_seed(1)
import random
random.seed(1)
import tensorflow as tf
import csv
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import pandas as pd
import tensorflow.keras.backend as K
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D, LeakyReLU, Add
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta, Adam, SGD, Nadam
from tensorflow.keras.regularizers import l1_l2, l1, l2
#from tensorflow_addons.layers import GroupNormalization, WeightNormalization
from sklearn.metrics import confusion_matrix
from datagenerator_pd import DataGenerator
from tensorflow.keras.utils import to_categorical
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'



#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-s', type=str, help='model name to save')
parser.add_argument('-c', type=int, help='number of cycles')
args = parser.parse_args()


params = {'batch_size': 5,
        'imagex': 160,
        'imagey': 192,
        'imagez': 160
        }

def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer[0])
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool4")(x)
    x=ReLU()(x)

    #block 5
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv5")(x)
    x=BatchNormalization(name="norm5")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),padding='same',name="maxpool5")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)

    #block 7, different from paper
    x=AveragePooling3D()(x)
    x=Dropout(.2)(x)
    x = Flatten(name="flat1")(x)
    x=Dense(units=1, activation='sigmoid',name="dense1")(x)


    return x



    
def compile_model():
    opt = Adam(lr=0.0001)
    metr = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]
    inputA = Input(shape=(params['imagex'], params['imagey'], params['imagez'], 1), name="InputA")
    z = sfcn([inputA])
    model = Model(inputs=[inputA], outputs=[z])
    model.summary()
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt, metrics=metr)
    return model
    
def scheduler(epoch, lr):
    return lr * tf.math.exp(-0.1)            


from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


model = compile_model()

fn_train = "train_pd_complete_adni.csv"
train = pd.read_csv(fn_train)
studies =  train['Study'].unique()
np.random.seed(42)  
np.random.shuffle(studies)


cycles = args.c
epoch_n = 5 #changed per experiment
for c in range(cycles):
    np.random.seed(42+c)  #comment this line for fixed order
    np.random.shuffle(studies) #comment this line for fixed order
    print("cycle-"+str(c))
    for s in studies:
        print("study-"+str(s))
        train_aux =  train[train['Study']==s]
        IDs_list = train_aux['Subject'].to_numpy()
        train_IDs = IDs_list
        batch_size=5
        if(len(train_IDs)<batch_size):
            batch_size=len(train_IDs)
        
        training_generator = DataGenerator(train_IDs,batch_size,(params['imagex'], params['imagey'], params['imagez']),True,fn_train,'Group_bin')
        lr_callback = tf.keras.callbacks.LearningRateScheduler(scheduler,verbose=1)
        history = model.fit(training_generator, epochs=int(epoch_n), callbacks=[lr_callback],verbose=2)

        history_dict = history.history

#model save
model.save(args.s+"_"+str(c)+".h5")


from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)
