import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout, BatchNormalization
from tensorflow.keras import layers
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from sklearn.cluster import MiniBatchKMeans
from sklearn.model_selection import cross_validate, train_test_split
from keras.regularizers import l2


training_data = np.loadtxt("training_data.txt")
training_data = training_data.reshape(training_data.shape[0], 64, 64, 1)

training_labels = np.loadtxt("training_labels.txt")
training_labels = to_categorical(training_labels)

testing_data = np.loadtxt("testing_data.txt")
testing_data = testing_data.reshape(testing_data.shape[0], 64, 64, 1)

testing_labels = np.loadtxt("testing_labels.txt")
testing_labels = to_categorical(testing_labels)

#define evaluation custom metric
def top_3_accuracy(y_true, y_pred):
        return keras.metrics.top_k_categorical_accuracy(y_true, y_pred, k=3)

#start building the model
model = Sequential()
model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(64,64,1)))
model.add(Dropout(.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.1))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.1))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(.1))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(.1))
model.add(Flatten())
model.add(Dense(1000, activation = 'relu', kernel_regularizer= l2(.000000000001)))
model.add(Dropout(.1))
model.add(Dense(29, activation='softmax'))
model.compile(loss='categorical_crossentropy',  optimizer='adam', metrics=['accuracy', top_3_accuracy])
print("model built")
model.summary()

#fit the models
history = model.fit(training_data, training_labels, batch_size=100, epochs=10, validation_data=(testing_data, testing_labels), shuffle=True)
print("MODEL TRAINING COMPLETE!")
model.save("finalModelRegularized")
print("MODEL SAVED")
                                 