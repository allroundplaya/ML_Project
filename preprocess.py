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
        print("Image has been resized !")

    else:
        print("Image is already a square, it has not been resized !")

    if width != IMG_SIZE:
        image = image.resize((IMG_SIZE, IMG_SIZE), Image.ANTIALIAS)

    # plt.imshow(image)
    # plt.show()

    return np.array(image)


def preprocess_all_img():
    if not os.path.exists('./data/train_data.npy'):
        if not os.path.exists('./paths/train_paths.npy'):
            train_paths = np.array([])

            for LETTER in LETTERS:
                for i in range(1, 2101):
                    train_paths = np.append(train_paths,
                                            './asl_alphabet_train/' + LETTER + '/' + LETTER + str(i) + '.jpg')
                    print(i)

            np.save("./paths/train_paths", train_paths)

        train_paths = np.load('./paths/train_paths.npy')
        train_data = np.array([])
        for path in train_paths:
            train_data = np.append(train_data, Reformat_Image(path))
        np.save('./data/train_data', train_data)

    if not os.path.exists('./data/test_data.npy'):
        if not os.path.exists('./paths/test_paths.npy'):
            test_paths = np.array([])

            for LETTER in LETTERS:
                for i in range(2101, 3001):
                    test_paths = np.append(test_paths,
                                           './asl_alphabet_train/' + LETTER + '/' + LETTER + str(i) + '.jpg')
                    print(i)

            np.save("./paths/test_paths", test_paths)

        test_paths = np.load('./paths/test_paths.npy')
        test_data = np.array([])
        for path in test_paths:
            test_data = np.append(test_data, Reformat_Image(path))
        np.save('./data/test_data', test_data)


preprocess_all_img()
