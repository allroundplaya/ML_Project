from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os

import time
from matplotlib import pyplot as plt

import datetime
from settings import *
batch_size = 14
num_classes = 29
epochs = 6


# input image dimensions
img_rows, img_cols = 200, 200

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = (np.load("./data/train_X.npz")['a'], np.load("./data/train_y.npz")['a']), \
                                       (np.load("./data/test_X.npz")['a'], np.load("./data/test_y.npz")['a'])

print(y_test)

"""
One-hot-encoding
"""
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)

for item in y_train:
    print(item)
print("train 끝")
for item in y_test:
    print(item)
print("test 끝")
y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)



print(y_test.shape)
print(y_test)


if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], x_train.shape[3], img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[3], img_cols)
    input_shape = (3, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, x_train.shape[3])
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, x_test.shape[3])
    input_shape = (img_rows, img_cols, 3)

# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(128, activation='relu'))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

if not os.path.exists('./model'):
    os.mkdir('./model')

dt = datetime.datetime.now()
dt.strftime("%Y%m%d_%H:%M")
model.save("model_"+dt.strftime("%Y%m%d_%H%M")+".h5")
