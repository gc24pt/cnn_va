import glob
import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint
import pickle

def load(path1, path2):
    images = []
    for file in natsorted(glob.glob(path1)):
        image = cv2.imread(file)
        images.append(image)
    target = pd.read_csv(path2, sep='\s+', header=None, index_col=0)
    return np.array(images), np.array(target)

def splitDataset(data, target):
    xtrain, xrem, ytrain, yrem = train_test_split(data, target, train_size=0.6, random_state=42)
    xval, xtest, yval, ytest = train_test_split(xrem, yrem, test_size=0.5, random_state=42)
    return xtrain, ytrain, xval, yval, xtest, ytest

def create_model(height,width):
    model = Sequential()
    model.add(Input(shape=(height,width,3)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(128, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Conv2D(256, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(2, padding='same'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(1000, activation='relu'))
    model.add(Dense(500, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='linear'))
    return model

def train_model(xtrain, ytrain, xval, yval, height, width):
    NUM_EPOCHS = 300
    BS = 32
    opt = Adam(learning_rate=0.0001)
    
    model = create_model(height, width)
    model.compile(optimizer=opt, loss='mse')
    
    checkpoint_filepath = 'checkpoint'
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
    hist = model.fit(xtrain, ytrain, validation_data=(xval,yval), batch_size=BS, epochs=NUM_EPOCHS, callbacks=[model_checkpoint_callback])
    return hist

def run(path, height, width):
    '''
    :param str path: folder where the input images (spectrograms) and target values (frequencies) are located
    :param int height: height of the input image
    :param int width: width of the input image
    '''
    
    data,target = load(path + 'specs/*.png', path + 'target.txt')
    xtrain, ytrain, xval, yval, xtest, ytest = splitDataset(data, target)
    hist = train_model(xtrain, ytrain, xval, yval, height, width)
    with open(path + 'history.pkl', 'wb') as f:
        pickle.dump(hist.history, f)

#run('Dados4_400s/', 12, 2401)
#run('Dados4_200s/', 12, 1201)
#run('Dados16_200s/', 16, 801)
