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
    if not os.path.exists("./paths"):
        os.mkdir("./paths")
    print("Preprocessing ", char + "(train)")
    if not os.path.exists('./data/train_X_' + char + '.npz'):
        if not os.path.exists('./paths/train_paths_' + char + '.npy'):
            train_paths = np.array([])
            for i in range(1, 15): # 나중에 2101로 바꾼다.
                train_paths = np.append(train_paths,
                                        './please/' + char + '/' + char + str(i) + '.jpg')
                #나중에 asl_alphabet_train으로 바꾼다.
            np.save("./paths/train_paths_"+char, train_paths)
            print("train path for " + char + " complete")

        train_paths = np.load('./paths/train_paths_' + char + '.npy')
        train_X = []
        train_y = np.array([])
        for path in train_paths:
            train_X.append(Reformat_Image(path))
            train_y = np.append(train_y, char)
        train_X = np.array(train_X)
        np.savez_compressed('./data/train_X_' + char, a=train_X)
        np.savez_compressed('./data/train_y_' + char, a=train_y)
        print("done!!!")


def preprocess_test_data(char):
    if not os.path.exists("./paths"):
        os.mkdir("./paths")
    print("Preprocessing " + char + "(test)")
    if not os.path.exists('./data/test_X_' + char + '.npz'):
        if not os.path.exists('./paths/test_paths_' + char + '.npy'):
            test_paths = np.array([])
            for i in range(15, 21): # 나중에 2101 3001로 바꾼다.
                test_paths = np.append(test_paths,
                                       './please/' + char + '/' + char + str(i) + '.jpg')
                # 나중에 asl_alphabet_train으로 바꾼다.
            np.save("./paths/test_paths_" + char, test_paths)
            print("test path for " + char + " complete")

        test_paths = np.load('./paths/test_paths_' + char + '.npy')
        test_X = []
        test_y = np.array([])
        for path in test_paths:
            test_X.append(Reformat_Image(path))
            test_y = np.append(test_y, char)
        test_X = np.array(test_X)
        np.savez_compressed('./data/test_X_' + char, a=test_X)
        np.savez_compressed('./data/test_y_' + char, a=test_y)
        print("done!!!")

def preprocess_all_data():
    if not os.path.exists("./data"):
        os.mkdir("./data")
    for LETTER in LETTERS:
        preprocess_train_data(LETTER)
        preprocess_test_data(LETTER)

    # train_X to npz
    np_tuple = []
    for letter in LETTERS:
        np_tuple.append(np.load('./data/train_X_'+letter+'.npz')['a'])
        print(letter, " finished")
    np_tuple = tuple(np_tuple)
    x_train = np.concatenate(np_tuple)
    np.savez_compressed("./data/train_X", a=x_train)

    # train_y to npz
    y_train = np.array([])
    for letter in LETTERS:
        y_train = np.append(y_train, np.load('./data/train_y_'+letter+'.npz')['a'])
        print(letter, " finished")
    np.savez_compressed("./data/train_y", a=y_train)


    # test_X to npz
    np_tuple = []
    for letter in LETTERS:
        np_tuple.append(np.load('./data/test_X_'+letter+'.npz')['a'])
        print(letter, " finished")
    np_tuple = tuple(np_tuple)
    x_test = np.concatenate(np_tuple)
    np.savez_compressed("./data/test_X", a=x_test)

    # test_y to npz
    y_test = np.array([])
    for letter in LETTERS:
        y_test = np.append(y_test, np.load('./data/test_y_'+letter+'.npz')['a'])
        print(letter, " finished")
    np.savez_compressed("./data/test_y", a=y_test)

    for LETTER in LETTERS:
        if os.path.exists('./data/train_X_'+LETTER+'.npz'):
            os.remove('./data/train_X_'+LETTER+'.npz')
        if os.path.exists('./data/train_y_' + LETTER + '.npz'):
            os.remove('./data/train_y_' + LETTER + '.npz')
        if os.path.exists('./data/test_X_' + LETTER + '.npz'):
            os.remove('./data/test_X_' + LETTER + '.npz')
        if os.path.exists('./data/test_y_' + LETTER + '.npz'):
            os.remove('./data/test_y_' + LETTER + '.npz')


preprocess_all_data()
