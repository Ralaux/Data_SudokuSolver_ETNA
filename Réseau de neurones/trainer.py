import os
os.environ['TF_ENABLE_CPU_OPTIMIZATIONS'] = '1'

import tensorflow as tf

import numpy as np
import pandas as pd
import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.models import Sequential
from keras.utils import Sequence
from keras.layers import *
import matplotlib.pyplot as plt

path = ""
data = pd.read_csv(path+"sudoku.csv")

try:
    data = pd.DataFrame({"quizzes":data["puzzle"],"solutions":data["solution"]})
except:
    pass

print(data.head())

print("Quiz:\n",np.array(list(map(int,list(data['quizzes'][0])))).reshape(9,9))
print("Solution:\n",np.array(list(map(int,list(data['solutions'][0])))).reshape(9,9))

#Utility Functions
class DataGenerator(Sequence):
    def __init__(self, df,batch_size = 16,subset = "train",shuffle = False, info={}):
        super().__init__()
        self.df = df
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subset = subset
        self.info = info
        
        self.data_path = path
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.floor(len(self.df)/self.batch_size))
    def on_epoch_end(self):
        self.indexes = np.arange(len(self.df))
        if self.shuffle==True:
            np.random.shuffle(self.indexes)
            
    def __getitem__(self,index):
        X = np.empty((self.batch_size, 9,9,1))
        y = np.empty((self.batch_size,81,1))
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        for i,f in enumerate(self.df['quizzes'].iloc[indexes]):
            self.info[index*self.batch_size+i]=f
            X[i,] = (np.array(list(map(int,list(f)))).reshape((9,9,1))/9)-0.5
        if self.subset == 'train': 
            for i,f in enumerate(self.df['solutions'].iloc[indexes]):
                self.info[index*self.batch_size+i]=f
                y[i,] = np.array(list(map(int,list(f)))).reshape((81,1)) - 1
        if self.subset == 'train': return X, y
        else: return X

model = Sequential()

model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same', input_shape=(9,9,1)))
model.add(BatchNormalization())
model.add(Conv2D(64, kernel_size=(3,3), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(Conv2D(128, kernel_size=(1,1), activation='relu', padding='same'))

model.add(Flatten())
model.add(Dense(81*9))
model.add(Reshape((-1, 9)))
model.add(Activation('softmax'))

adam = keras.optimizers.Adam(lr=.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

model.summary()

train_idx = int(len(data)*0.95)
data = data.sample(frac=1).reset_index(drop=True)
training_generator = DataGenerator(data.iloc[:train_idx], subset = "train", batch_size=640)
validation_generator = DataGenerator(data.iloc[train_idx:], subset = "train",  batch_size=640)

training_generator.__getitem__(4)[0].shape

from keras.callbacks import Callback, ModelCheckpoint, ReduceLROnPlateau
filepath1="weights-improvement-{epoch:02d}-{val_accuracy:.2f}.hdf5"
filepath2 = "best_weights.hdf5"
checkpoint1 = ModelCheckpoint(filepath1, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
checkpoint2 = ModelCheckpoint(filepath2, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    patience=3,
    verbose=1,
    min_lr=1e-6
)
callbacks_list = [checkpoint1,checkpoint2,reduce_lr]

history = model.fit_generator(training_generator, validation_data = validation_generator, epochs = 1, verbose=1,callbacks=callbacks_list )

model.load_weights('best_weights.hdf5')