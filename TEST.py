from __future__ import print_function
import keras

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import numpy as np
from settings import *
batch_size = 100
num_classes = 29
epochs = 12


# input image dimensions
img_rows, img_cols = 200, 200

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = (np.load("./data/train_X.npz")['a'], np.load("./data/train_y.npz")['a']), \
                                       (np.load("./data/test_X.npz")['a'], np.load("./data/test_y.npz")['a'])

# for letter in LETTERS:
#     if letter == 'A':
#         continue
#     x_train = np.concatenate((x_train, np.load('./data/train_X_'+letter+'.npy')))
#     x_test = np.concatenate((x_test, np.load('./data/test_X_' + letter + '.npy')))
#     y_train = np.concatenate((y_train, np.load('./data/train_y_' + letter + '.npy')))
#     y_test = np.concatenate((y_test, np.load('./data/test_y_' + letter + '.npy')))
#     print(letter, " has been finished")

label_encoder_1 = LabelEncoder()
label_encoder_2 = LabelEncoder()
integer_encoded_1 = label_encoder_1.fit_transform(y_train)
integer_encoded_2 = label_encoder_2.fit_transform(y_test)
onehot_encoder_1 = OneHotEncoder(sparse=False)
onehot_encoder_2 = OneHotEncoder(sparse=False)
integer_encoded_1 = integer_encoded_1.reshape(len(integer_encoded_1), 1)
integer_encoded_2 = integer_encoded_2.reshape(len(integer_encoded_2), 1)
y_train = onehot_encoder_1.fit_transform(integer_encoded_1)
y_test = onehot_encoder_2.fit_transform(integer_encoded_2)

print(y_test.shape)
print(y_train.shape)


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
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
