from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import LabelEncoder
import numpy as np
import os
import datetime
from settings import *

class CNN:
    def __init__(self):

        # input image dimensions
        self.img_rows = 200
        self.img_cols = 200
        self.num_classes = 29
        # the data, split between train and test sets
        (self.x_train, self.y_train), (self.x_test, self.y_test) = (None, None), (None, None)
        self.input_shape = (self.img_rows, self.img_cols, 3)
        self.model = Sequential()
        self.score = None

    def load_dataset(self, x_train_path="./temp_data/train_X.npz", y_train_path="./temp_data/train_y.npz",
                     x_test_path="./temp_data/test_X.npz", y_test_path="./temp_data/test_y.npz"):
        (self.x_train, self.y_train), (self.x_test, self.y_test) = \
            (np.load(x_train_path)['a'], np.load(y_train_path)['a']), \
            (np.load(x_test_path)['a'], np.load(y_test_path)['a'])

    def make_cnn_model(self):

        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(2, 2), activation='relu', input_shape=self.input_shape))
#        self.model.add(Conv2D(32, kernel_size=(3, 3), strides=(2,2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
#        self.model.add(Dropout(0.1))
        print()

        self.model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
#        self.model.add(Dropout(0.1))
        print()

        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.num_classes, activation='softmax'))
        print()

        self.model.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        keras.utils.print_summary(self.model)

    def train_model(self, batch_size, epochs):

        # normalizing x
        x_train_normalized = self.x_train.astype('float32')/255
        x_test_normalized = self.x_test.astype('float32')/255
        print("x_train shape: ", x_train_normalized.shape)
        print("x_test shape: ", x_test_normalized.shape)

        # one-hot encoding labels
        label_encoder = LabelEncoder()
        y_train_onehot_encoded = keras.utils.to_categorical(label_encoder.fit_transform(self.y_train),
                                                            num_classes=self.num_classes)
        y_test_onehot_encoded = keras.utils.to_categorical(label_encoder.fit_transform(self.y_test),
                                                           num_classes=self.num_classes)
        print("y_train shape: ", y_train_onehot_encoded.shape)
        print("y_test shape: ", y_test_onehot_encoded.shape)

        self.model.fit(x_train_normalized, y_train_onehot_encoded,
                  batch_size=batch_size,
                  epochs=epochs,
                  verbose=2,  # 진행상태를 보겠다는 의미입니다. 0: 아무것도 안보여줌, 1: 진행상태 bar 표시, 2: 한 epoch 당 한 줄
                  validation_data=(x_test_normalized, y_test_onehot_encoded))
        self.score = self.model.evaluate(x_test_normalized, y_test_onehot_encoded, verbose=0)

    def save_model(self):
        if not os.path.exists('./model'):
            os.mkdir('./model')

        dt = datetime.datetime.now()
        dt.strftime("%Y%m%d_%H:%M")

        if not os.path.exists("./model/model_" + dt.strftime("%Y%m%d_%H%M") + ".h5"):
            self.model.save("./model/model_" + dt.strftime("%Y%m%d_%H%M") + ".h5")

    def load_model(self, path):
        self.model = keras.models.loadmodel(path)
    def print_score(self):
        print('=' * 30)
        print('Test Loss: %.4f' % (self.score[0]))
        print('Test Accuracy: %.4f' % (self.score[1]))
        print('=' * 30)
#
# cnn = CNN()
# cnn.make_cnn_model()