import glob
import cv2
import numpy as np
from natsort import natsorted
from tensorflow.keras.models import load_model

def load(path):
    images = []
    for file in natsorted(glob.glob(path)):
        image = cv2.imread(file)
        images.append(image)
    return np.array(images)

def predict(path):
    data = load(path + 'specs/*.png')
    model = load_model('model/')
    ypred = model.predict(data)
    return ypred

#ypred = predict('test/')
