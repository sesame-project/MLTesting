
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras import utils

import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import numpy as np
# import matplotlib.pyplot as plt
# import seaborn as sns
from scipy.io import loadmat
# from skimage import color
# from skimage import io
from tensorflow.keras.datasets import cifar10
from sklearn.model_selection import train_test_split
from scipy.io import loadmat
import tensorflow as tf
import numpy as np

def split_data(split, X, Y):
    per=1-split
    # x_test, x_val, y_test, y_val = train_test_split(X,Y, test_size=per, shuffle=False)  # 0.25 x 0.8 = 0.2
    x_test = X[:int(len(X)*per)]
    y_test=Y[:int(len(X)*per)]
    return x_test, y_test

def load_MNIST_32():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_test, X_val, y_test, y_val = train_test_split(x_test, y_test, test_size=0.64,
                                                    random_state=1)  # 0.25 x 0.8 = 0.2

    # #expand new axis, channel axis
    x_train = np.expand_dims(x_train, axis=-1)
    x_test = np.expand_dims(x_test, axis=-1)
    X_val = np.expand_dims(x_test, axis=-1)
    # [optional]: we may need 3 channel (instead of 1)
    x_train = np.repeat(x_train, 3, axis=-1)
    x_test = np.repeat(x_test, 3, axis=-1)
    X_val = np.repeat(x_test, 3, axis=-1)
    # it's always better to normalize
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    X_val = x_test.astype('float32') / 255
    # resize the input shape , i.e. old shape: 28, new shape: 32
    x_train = tf.image.resize(x_train, [32, 32])  # if we want to resize
    x_test = tf.image.resize(x_test, [32, 32])
    X_val = tf.image.resize(x_test, [32, 32])
    # one hot
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
    y_val = tf.keras.utils.to_categorical(y_test, num_classes=10)


    return x_train, y_train, x_test, y_test,X_val, y_val

def load_SVHN_(typedt):
    train_raw = loadmat('dataset/SVHN/train_32x32.mat')
    test_raw = loadmat('dataset/SVHN/test_32x32.mat')
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])
    # print("DATA SVHN is loaded")

    train_labels = train_raw['y']
    test_labels = test_raw['y']
    # Fix the axes of the images

    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    # print(train_images.shape)
    # print(test_images.shape)
    # Convert train and test images into 'float64' type

    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')
    train_images /= 255.0
    test_images /= 255.0
    X_train = train_images
    y_train = train_labels
    X_test = test_images
    y_test = test_labels
    # print("DATA SVHN is Transformed")
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                      test_size=0.05, random_state=1)
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)



    return X_train, y_train, X_test, y_test,X_val, y_val

def SVHN_data(typedt,channel_first=True):
    train_raw = loadmat('dataset/SVHN/train_32x32.mat')
    test_raw = loadmat('dataset/SVHN/test_32x32.mat')
    train_images = np.array(train_raw['X'])
    test_images = np.array(test_raw['X'])
    print("DATA SVHN is loaded")

    train_labels = train_raw['y']
    test_labels = test_raw['y']
    # Fix the axes of the images

    train_images = np.moveaxis(train_images, -1, 0)
    test_images = np.moveaxis(test_images, -1, 0)
    lb = LabelBinarizer()
    train_labels = lb.fit_transform(train_labels)
    test_labels = lb.fit_transform(test_labels)

    # print(train_images.shape)
    # print(test_images.shape)
    # Convert train and test images into 'float64' type

    train_images = train_images.astype('float64')
    test_images = test_images.astype('float64')
    train_images /= 255.0
    test_images /= 255.0
    X_train = train_images
    y_train = train_labels
    X_test = test_images
    y_test = test_labels
    print("DATA SVHN is Transformed")
    X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels,
                                                      test_size=0.05, random_state=1)
    # print(X_train.shape)
    # print(X_val.shape)
    # print(X_test.shape)

    # if channel_first:
    #     X_train = X_train.reshape(X_train.shape[0], 1, 32, 32)
    #
    #     X_test = X_test.reshape(X_test.shape[0], 1, 32, 32)

    return X_train, y_train, X_test, y_test,X_val, y_val


def load_EMNIST(one_hot=True, channel_first=True):
    testing_letter = pd.read_csv('dataset/emnist/emnist-letters-test.csv')
    training_letter = pd.read_csv('dataset/emnist/emnist-letters-train.csv')
    # print(training_letter.shape)
    # print(testing_letter.shape)
    # training_letters
    y1 = np.array(training_letter.iloc[:, 0].values)
    x1 = np.array(training_letter.iloc[:, 1:].values)
    # testing_labels
    y2 = np.array(testing_letter.iloc[:, 0].values)
    x2 = np.array(testing_letter.iloc[:, 1:].values)
    # print(y1.shape)
    # print(x1.shape)
    # Normalise and reshape data
    train_images = x1 / 255.0
    test_images = x2 / 255.0

    train_images_number = train_images.shape[0]
    train_images_height = 28
    train_images_width = 28
    train_images_size = train_images_height * train_images_width

    train_images = train_images.reshape(train_images_number, train_images_height, train_images_width, 1)

    test_images_number = test_images.shape[0]
    test_images_height = 28
    test_images_width = 28
    test_images_size = test_images_height * test_images_width

    test_images = test_images.reshape(test_images_number, test_images_height, test_images_width, 1)
    # Transform labels
    number_of_classes = 37

    y1 = tf.keras.utils.to_categorical(y1, number_of_classes)
    y2 = tf.keras.utils.to_categorical(y2, number_of_classes)
    train_x, test_x, train_y, test_y = train_test_split(train_images, y1, test_size=0.1, random_state=42)

    return train_x, train_y, test_x, test_y





def load_CIFAR(one_hot=True):
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05, random_state=1) # 0.25 x 0.8 = 0.2

    if one_hot:
        y_train = utils.to_categorical(y_train, num_classes=10)
        y_test = utils.to_categorical(y_test, num_classes=10)
        y_val= utils.to_categorical(y_test, num_classes=10)

    return X_train, y_train, X_test, y_test, X_val, y_val



def load_MNIST(one_hot=True, channel_first=True):
    """
    Load MNIST data
    :param channel_first:
    :param one_hot:
    :return:
    """
    # Load data
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Preprocess dataset
    # Normalization and reshaping of input.
    if channel_first:
        X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)

        X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)

    else:
        # X_train = X_train.astype('float32').reshape(-1, 32, 32, 1)
        # X_test = X_test.astype('float32').reshape(-1, 32, 32, 1)
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)


    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255

    if one_hot:
        # For output, it is important to change number to one-hot vector.
        y_train = utils.to_categorical(y_train, num_classes=10)
        y_test = utils.to_categorical(y_test, num_classes=10)


    return X_train, y_train, X_test, y_test
def load_MNISTVAL(typedt, one_hot=True, channel_first=True):
        """
        Load MNIST data
        :param channel_first:
        :param one_hot:
        :return:
        """
        if typedt=="fashion":
          # Load data
          (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()#fashion_mnist.load_data()
          X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1) # 0.25 x 0.8 = 0.2
        # Load data
        elif  typedt=="ini":
          (X_train, y_train), (X_test, y_test) = mnist.load_data()
          print ("Loading....")
          X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=1) # 0.25 x 0.8 = 0.2

        # Preprocess dataset
        # Normalization and reshaping of input.
        if channel_first:
            X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
            X_test = X_test.reshape(X_test.shape[0], 1, 28, 28)
            X_val= X_val.reshape(X_val.shape[0], 1, 28, 28)
        else:
            X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
            X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
            X_val =  X_val.reshape(X_val.shape[0], 28, 28, 1)

        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        X_val = X_val.astype('float32')
        X_train /= 255
        X_test /= 255
        X_val /= 255

        if one_hot:
            # For output, it is important to change number to one-hot vector.
            y_train = utils.to_categorical(y_train, num_classes=10)
            y_test = utils.to_categorical(y_test, num_classes=10)
            y_val = utils.to_categorical(y_val, num_classes=10)

        return X_train, y_train, X_test, y_test, X_val, y_val
def load_CIFARVAL(typedt, one_hot=True):
    if typedt == "zeroshot":
        # Load data
        (X_train, y_train), (X_test, y_test) =cifar100.load_data(label_mode="fine")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05,
                                                          random_state=1)  # 0.25 x 0.8 = 0.2
        if one_hot:
            y_train = utils.to_categorical(y_train, num_classes=100)
            y_test = utils.to_categorical(y_test, num_classes=100)
            y_val = utils.to_categorical(y_test, num_classes=100)

    elif typedt == "ini":
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        print("Loading....")
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.05,
                                                          random_state=1)  # 0.25 x 0.8 = 0.2
        if one_hot:
            y_train = utils.to_categorical(y_train, num_classes=10)
            y_test = utils.to_categorical(y_test, num_classes=10)
            y_val = utils.to_categorical(y_val, num_classes=10)
        # print("from load!!!!!")
        # print(X_val.shape, y_val.shape)



    return X_train, y_train, X_test, y_test, X_val, y_val
def load_driving_data(path='driving_data/', batch_size=64, shape=(100, 100)):
    xs = []
    ys = []
    start_load_time = time.time()
    with open(path + 'final_example.csv', 'r') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            xs.append(path + 'center/' + line.split(',')[0] + '.jpg')
            ys.append(float(line.split(',')[1]))
    # shuffle list of images
    c = list(zip(xs, ys))
    random.shuffle(c)
    xs, ys = zip(*c)

    train_xs = xs
    train_ys = ys

    return train_xs, train_ys
def filter_val_set(desired_class, X, Y):
    """
    Filter the given sets and return only those that match the desired_class value
    :param desired_class:
    :param X:
    :param Y:
    :return:
    """
    X_class = []
    Y_class = []
    for x, y in zip(X, Y):
        if y[desired_class] == 1:
            X_class.append(x)
            Y_class.append(y)
    # print("Validation set filtered for desired class: " + str(desired_class))
    return np.array(X_class), np.array(Y_class)
