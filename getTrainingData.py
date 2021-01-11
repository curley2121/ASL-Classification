import os
import cv2
import numpy as np

from sklearn.cluster import MiniBatchKMeans

from keras.preprocessing.image import random_shift, random_zoom

from sklearn.model_selection import cross_validate, train_test_split

def load_training():
    labels_dict = {'A':0,'B':1,'C':2,'D':3,'E':4,'F':5,'G':6,'H':7,'I':8,'J':9,'K':10,'L':11,'M':12,
                   'N':13,'O':14,'P':15,'Q':16,'R':17,'S':18,'T':19,'U':20,'V':21,'W':22,'X':23,'Y':24,
                   'Z':25,'space':26,'del':27,'nothing':28}
    data = []
    targets = []
    size = 64, 64
    print("reading in training data...")
    for folder in os.listdir("set1/asl_alphabet_train/"):
        print("First set; reading in training data for: " + folder)
        count = 0
        for image in os.listdir("set1/asl_alphabet_train/" + folder):
            count += 1
            img = cv2.imread("set1/asl_alphabet_train/" + folder + "/" + image)
            img = cv2.resize(img, size)
            #randomization goes here
            img = random_shift(img, .2, .2, fill_mode = "reflect", row_axis = 0, col_axis = 1, channel_axis = 2)
            img = random_zoom(img, zoom_range=[.85, .85], row_axis = 0, col_axis = 1, channel_axis = 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            img = convertBW(img)
            data.append(img)
            targets.append(labels_dict[folder])
        print("First set; there were " + str(count) + "images in folder " + folder)
    for folder in os.listdir("set2/ASLDataSet/"):
        print("Second set; reading in training data for: " + folder)
        count = 0
        for image in os.listdir("set2/ASLDataSet/" + folder):
            count += 1
            img = cv2.imread("set2/ASLDataSet/" + folder + "/" + image)
            img = cv2.resize(img, size)
            #randomization goes here
            img = random_shift(img, .2, .2, fill_mode = "reflect", row_axis = 0, col_axis = 1, channel_axis = 2)
            img = random_zoom(img, zoom_range=[.85, .85], row_axis = 0, col_axis = 1, channel_axis = 2)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            img = convertBW(img)
            data.append(img)
            targets.append(labels_dict[folder])
        print("Second set; there were " + str(count) + "images in folder " + folder)
    data = np.array(data)
    return data, targets


def convertBW(img):
    reshaped = np.reshape(img[:, :, :], (-1, 3))
    kmeans = MiniBatchKMeans(n_clusters=2, n_init=25, max_iter=200).fit(reshaped)
    clustering = np.reshape(np.array(kmeans.labels_, dtype=np.uint8),
            (img.shape[0], img.shape[1], 1))
    return clustering


data, targets = load_training()
#reshaped = data.reshape(data.shape[0], -1)
print(f"There were {data.shape[0]} images.")


training_data, testing_data, training_labels, testing_labels = train_test_split(data, targets, test_size = 0.4)
validation_data, testing_data, validation_labels, testing_labels = train_test_split(testing_data, testing_labels, test_size = 0.5)

reshaped_train = training_data.reshape(training_data.shape[0], -1)
np.savetxt("training_data.txt", reshaped_train)
np.savetxt("training_labels.txt", training_labels)
print("The training labels and data have been read in and saved to files.")

reshaped_val = validation_data.reshape(validation_data.shape[0], -1)
np.savetxt("val_data.txt", reshaped_val)
np.savetxt("val_labels.txt", validation_labels)
print("The validation labels and data have been read in and saved to files.")

reshaped_test = testing_data.reshape(testing_data.shape[0], -1)
np.savetxt("testing_data.txt", reshaped_test)
np.savetxt("testing_labels.txt", testing_labels)
print("The testing labels and data have been read in and saved to files.")



