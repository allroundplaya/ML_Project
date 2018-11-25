from settings import *
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np


def Reformat_Image(ImageFilePath):
    image = Image.open(ImageFilePath, 'r')
    image_size = image.size
    width = image_size[0]
    height = image_size[1]

    if(width != height):
        bigside = width if width > height else height

        background = Image.new('RGB', (bigside, bigside), (255, 255, 255))
        offset = (int(round(((bigside - width)/2), 0)), int(round(((bigside - height)/2),0)))

        background.paste(image, offset)
        background.save('out.png')
        plt.imshow(background)
        plt.show()
        print(type(background))
        print("Image has been resized !")
        return background

    else:
        print("Image is already a square, it has not been resized !")

        plt.imshow(image)
        plt.show()
        print(type(image))
        return image








def preprocess_all_img():
    # image = tf.image.decode_and_crop_jpeg(tf.read_file(TRAIN_A_DIR + "A1.jpg"), channels=3)

    # image = Image.open(TRAIN_A_DIR + "A1.jpg")
    # image = Image.open("china.jpg")
    # image = image.resize((256, 256))


    # Reformat_Image(TRAIN_A_DIR + "A1.jpg")
    Reformat_Image("china.jpg")
    image = tf.image.decode_image(tf.read_file(Reformat_Image(TRAIN_A_DIR + "A1.jpg")), channels=3)

    # plt.imshow(image)
    # plt.show()

    # #
    # with tf.Session() as sess:
    #     plt.imshow(sess.run(image))
    #     plt.show()


preprocess_all_img()




