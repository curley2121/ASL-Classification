import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import KMeans
#from sklearn.model_selection import cross_validate, train_test_split

from sklearn.cluster import MiniBatchKMeans
from keras.preprocessing.image import random_shift, random_zoom
from sklearn.model_selection import cross_validate, train_test_split
from PIL import Image, ImageOps

#importing custom metric from compilation
def top_3_accuracy(y_true, y_pred):
    return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

model = keras.models.load_model("../MLtermproject/finalModelRegularized", custom_objects={'top_3_accuracy':top_3_accuracy})

data = np.loadtxt("val_data.txt")
data = data.reshape(data.shape[0], 64, 64, 1)

labels = np.loadtxt("val_labels.txt")
labels = to_categorical(labels)

#predictions = model.predict(data)
#score = model.evaluate(data, labels)
#print("Accuracy: ", score[1])
#print("Top 3 Accuracy: ",score[2])
def convertBW(img):
    reshaped = np.reshape(img[:, :, :], (-1, 3))
    kmeans = MiniBatchKMeans(n_clusters=2, n_init=25, max_iter=200).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
        (img.shape[0], img.shape[1], 1))
    return clustering

letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'space', 'del', 'nothing']

size= 64, 64
for image in os.listdir("HandPics/"):
    img = cv2.imread("HandPics/" + image)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = convertBW(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    indices = prediction.argsort()[-3:][::-1]
    #print(str(image))
    #for num in indices: 
        #print(letters[num])

for image in os.listdir("HandPicsRound2/"):
    img = cv2.imread("HandPicsRound2/"+ image)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = convertBW(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    indices = prediction.argsort()[-3:][::-1]
    #print(str(image))
    #for num in indices:
        #print(letters[num])

for image in os.listdir("HandPics3/"):
    img = cv2.imread("HandPics3/"+ image)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = convertBW(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    indices = prediction.argsort()[-3:][::-1]
    #print(str(image),":")
    temp = []
    for num in indices:
        temp.append(letters[num])
    #print(temp)

for image in os.listdir("HandPics4/"):
    img = cv2.imread("HandPics4/"+ image)
    img = cv2.resize(img, size)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    img = convertBW(img)
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    indices = prediction.argsort()[-3:][::-1]
    print(str(image),":")
    temp = []
    for num in indices:
        temp.append(letters[num])
    print(temp)
