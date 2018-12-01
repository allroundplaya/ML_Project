from settings import *
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os

IMG_SIZE = 200

def Reformat_Image(path):
    """
    function for reformatting image files to (256, 256, 3) size np.array.
    A rectangular image can be reformatted keeping its aspect ratio with white padding.

    :param path: str. path of the image
    :return: np.array. 3D np.array with size (256, 256, 3)
    """
    image = Image.open(path, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if width != height:
        long = width if width > height else height
        background = Image.new('RGB', (long, long), (255, 255, 255))
        offset = (int(round(((long - width)/2), 0)), int(round(((long - height)/2), 0)))

        background.paste(image, offset)
        image = background
        image_size = image.size
        width = image_size[0]
        height = image_size[1]
        # print("Image has been resized !")

    # else:
        # print("Image is already a square, it has not been resized !")

    if width != IMG_SIZE:
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

    # plt.imshow(image)
    # plt.show()

    return np.array(image)


def preprocess_train_data(char):
    print("Preprocessing ", char + "(train)")
    if not os.path.exists('./data/train_X_' + char + '.npy'):
        if not os.path.exists('./paths/train_paths_' + char + '.npy'):
            train_paths = np.array([])
            for i in range(1, 2101):
                train_paths = np.append(train_paths,
                                        './asl_alphabet_train/' + char + '/' + char + str(i) + '.jpg')
            np.save("./paths/train_paths_"+char, train_paths)
            print("train path for " + char + " complete")

        train_paths = np.load('./paths/train_paths_' + char + '.npy')
        train_X = []
        train_y = np.array([])
        for path in train_paths:
            train_X.append(Reformat_Image(path))
            train_y = np.append(train_y, char)
        train_X = np.array(train_X)
        np.save('./data/train_X_' + char, train_X)
        np.save('./data/train_y_' + char, train_y)


def preprocess_test_data(char):
    print("Preprocessing " + char + "(test)")
    if not os.path.exists('./data/test_X_' + char + '.npy'):
        if not os.path.exists('./paths/test_paths_' + char + '.npy'):
            test_paths = np.array([])
            for i in range(2101, 3001):
                test_paths = np.append(test_paths,
                                       './asl_alphabet_train/' + char + '/' + char + str(i) + '.jpg')

            np.save("./paths/test_paths_" + char, test_paths)
            print("test path for " + char + " complete")

        test_paths = np.load('./paths/test_paths_' + char + '.npy')
        test_X = []
        test_y = np.array([])
        for path in test_paths:
            test_X.append(Reformat_Image(path))
            test_y = np.append(test_y, char)
        test_X = np.array(test_X)
        np.save('./data/test_X_' + char, test_X)
        np.save('./data/test_y_' + char, test_y)

def preprocess_all_data():
    for LETTER in LETTERS:
        preprocess_train_data(LETTER)
        preprocess_test_data(LETTER)
