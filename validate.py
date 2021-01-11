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
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.metrics import confusion_matrix

#importing custom metric from compilation
def top_3_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

model = keras.models.load_model("../MLtermproject/finalModelRegularized", custom_objects={'top_3_accuracy':top_3_accuracy})

data = np.loadtxt("val_data.txt")
data = data.reshape(data.shape[0], 64, 64, 1)

labels = np.loadtxt("val_labels.txt")
labels = to_categorical(labels)
predictions = model.predict(data)


print("labels shape = "+ str(labels.shape))
print(labels[0:2])
print("predictions shape = "+ str(predictions.shape))
print(labels[0:2])
score = model.evaluate(data, labels)
print("Accuracy: ", score[1])
print("Top 3 Accuracy: ",score[2])
labels = np.argmax(labels, axis=1)
predictions = np.argmax(predictions, axis=1)
confusion = confusion_matrix(labels, predictions)
print(confusion)
