import os
os.environ["KERAS_BACKEND"] = "torch"
import glob
import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten, Dropout
from keras.callbacks import ModelCheckpoint
import pickle

SPECS_PATH = "data/specs/*.png"
TARGET_PATH = "data/target.txt"
RESULTS_PATH = "exp1234/"

def load():
    images = []
    for file in natsorted(glob.glob(SPECS_PATH)):
        image = cv2.imread(file)
        images.append(image)
    target = pd.read_csv(TARGET_PATH, sep='\s+', header=None, index_col=0)
    return np.array(images), np.array(target)

def splitData(data, target):
    xtrain, xtemp, ytrain, ytemp = train_test_split(data, target, test_size=0.2, shuffle=False, random_state=42)
    xval, _, yval, _ = train_test_split(xtemp, ytemp, test_size=0.5, shuffle=False, random_state=42)
    return xtrain, ytrain, xval, yval

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
    NUM_EPOCHS = 20
    BS = 32
    opt = Adam(learning_rate=0.0001)
    
    model = create_model(height, width)
    model.compile(optimizer=opt, loss='mse')
    
    checkpoint_filepath = RESULTS_PATH + 'checkpoint.keras'
    model_checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=False,
    monitor='val_loss',
    mode='min',
    save_best_only=True)
    
    hist = model.fit(xtrain, ytrain, validation_data=(xval,yval), batch_size=BS, epochs=NUM_EPOCHS, callbacks=[model_checkpoint_callback])
    return hist

def run(height, width):
    '''
    :param int height: height of the input image
    :param int width: width of the input image
    '''
    
    data, target = load()
    xtrain, ytrain, xval, yval = splitData(data, target)
    hist = train_model(xtrain, ytrain, xval, yval, height, width)
    with open(RESULTS_PATH + 'history.pkl', 'wb') as f:
        pickle.dump(hist.history, f)

if __name__ == "__main__":
    if not os.path.exists(RESULTS_PATH):
        os.makedirs(RESULTS_PATH)
    run(16, 801)
