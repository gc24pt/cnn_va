import glob
import cv2
import numpy as np
from natsort import natsorted
from tensorflow.keras.models import load_model

def load(path1, path2):
    images = []
    for file in natsorted(glob.glob(path1)):
        image = cv2.imread(file)
        images.append(image)
    return np.array(images)

def predict(path):
    data = load(path + 'specs/*.png')
    model = load_model('checkpoint/')
    ypred = model.predict(data)
    return ypred

#ypred = predict('Dados16_200s/')
