import os
os.environ["KERAS_BACKEND"] = "torch"
import glob
import cv2
import numpy as np
import pandas as pd
from natsort import natsorted
from sklearn.model_selection import train_test_split
from keras.models import load_model
import plotly.graph_objects as go
import pickle
import plotly.io as pio
pio.renderers.default = 'browser'

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

def loadPickle(path):
    with open(path, 'rb') as f:
        pickleData = pickle.load(f)
    return pickleData

def testPlot(test, cnn):
    time = np.arange(0,len(test[0]))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=cnn[0], mode="markers", line_color='red', name="Predicted values"))
    fig.add_trace(go.Scatter(x=time, y=cnn[1], mode="markers", line_color='red', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=cnn[2], mode="markers", line_color='red', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=cnn[3], mode="markers", line_color='red', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=cnn[4], mode="markers", line_color='red', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=test[0], mode="markers", line_color='blue', name="Real values"))
    fig.add_trace(go.Scatter(x=time, y=test[1], mode="markers", line_color='blue', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=test[2], mode="markers", line_color='blue', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=test[3], mode="markers", line_color='blue', showlegend=False))
    fig.add_trace(go.Scatter(x=time, y=test[4], mode="markers", line_color='blue', showlegend=False))
    fig.update_layout(title_text="Real values vs CNN predicted values", xaxis_title="Test hour", yaxis_title="Frequency", font=dict(
        family="Open Sans",
        size=26,
        color="black"
    ))
    pio.write_image(fig, RESULTS_PATH + 'test.png', width=1920, height=1080)

def lossPlot(history):
    train_loss = history['loss']
    val_loss = history['val_loss']
    train_loss = np.log(train_loss)
    val_loss = np.log(val_loss)
    fig = go.Figure()
    epochs = np.arange(1,len(train_loss)+1)
    fig.add_trace(go.Scatter(x=epochs, y=train_loss, mode='lines', line_color='blue', name='Train loss'))
    fig.add_trace(go.Scatter(x=epochs, y=val_loss, mode='lines', line_color='red', name='Validation loss'))
    fig.update_layout(title_text="Model loss (log scale)", xaxis_title="Epoch", yaxis_title="log(Loss)", font=dict(
        family="Open Sans",
        size=26,
        color="black"
    ))
    pio.write_image(fig, RESULTS_PATH + 'loss.png', width=1920, height=1080)

def splitData(data, target):
    _, xtemp, _, ytemp = train_test_split(data, target, test_size=0.2, shuffle=False, random_state=42)
    _, xtest, _, ytest = train_test_split(xtemp, ytemp, test_size=0.5, shuffle=False, random_state=42)
    return xtest, ytest

#Predictions on the test set
def predict():
    data,target = load()
    xtest, ytest = splitData(data, target)
    model = load_model(RESULTS_PATH + 'checkpoint.keras')
    ypred = model.predict(xtest)
    results = []
    results.append(ypred)
    results.append(ytest)
    with open(RESULTS_PATH + 'results.pkl', 'wb') as f:
        pickle.dump(results, f)
    return results

def splitResults(ytest, ypred):
    test = [[] for _ in range(5)]
    cnn = [[] for _ in range(5)]
    for i,j in zip(ytest,ypred):
        test[0].append(i[0])
        test[1].append(i[1])
        test[2].append(i[2])
        test[3].append(i[3])
        test[4].append(i[4])
        cnn[0].append(j[0])
        cnn[1].append(j[1])
        cnn[2].append(j[2])
        cnn[3].append(j[3])
        cnn[4].append(j[4])
    return test, cnn

def run():
    results = predict()
    #results = loadPickle(RESULTS_PATH + 'results.pkl')
    ypred = results[0]
    ytest = results[1]
    test, cnn = splitResults(ytest, ypred)
    testPlot(test, cnn)
    history = loadPickle(RESULTS_PATH + 'history.pkl')
    lossPlot(history)
    val_loss = history['val_loss']
    print(np.min(val_loss))
    print(np.log(np.min(val_loss)))

if __name__ == "__main__":
    run()
