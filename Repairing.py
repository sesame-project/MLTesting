import os
import numpy as np
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
from tqdm import tqdm
import argparse
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from Dataprocessing import  load_coco, load_leaves_V, load_CIFARVAL

from sklearn.metrics import roc_auc_score
import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.applications.vgg19  import VGG19
from keras.preprocessing.image import ImageDataGenerator
# from coco_preprocess import *
from sklearn.metrics import classification_report
from sklearn.metrics import zero_one_loss
import tensorflow as tf
import pathlib
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from tensorflow.keras import utils
import tensorflow as tf

import tensorflow.keras.optimizers
import cv2
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import os
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import average_precision_score
from datetime import datetime
from tensorflow.keras.models import model_from_json, load_model, save_model
# import multiprocessing
# import itertools

def accuracy(y_true, y_pred):
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score

    """

    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0

    for yt, yp in zip(y_true, np.argmax(y_pred, axis=1)):

        if yt == np.argmax(yp)[0]:
            correct_predictions += 1

    # returns accuracy
    return correct_predictions / len(y_true)

def get_xy_generator_folder(train_generator, size_batch):
    train_generator.reset()
    X_train, y_train = next(train_generator)
    for i in tqdm(range(int((train_generator.samples)/size_batch))):
        img, label = next(train_generator)
        X_train = np.append(X_train, img, axis=0)
        y_train = np.append(y_train, label, axis=0)
    # print(X_train.shape, y_train.shape)
    return X_train,y_train
def get_coco_data():
    trdata = ImageDataGenerator()
    traindata = trdata.flow_from_directory(directory="/home/sondess/sm2672/DataAugmentation_pipline/dataset/coco/data_structured/train", target_size=(32, 32))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="/home/sondess/sm2672/DataAugmentation_pipline/dataset/coco/data_structured/test", target_size=(32, 32))
    vldata = ImageDataGenerator()
    valdata = vldata.flow_from_directory(directory="/home/sondess/sm2672/DataAugmentation_pipline/dataset/coco/data_structured/val", target_size=(32, 32))
    X_train, y_train = next(traindata)
    X_test, y_test = next(testdata)
    X_val, y_val = next(valdata)
    size = X_train.shape[0]
    sizetest=X_test.shape[0]
    sizeval = X_val.shape[0]
    X_train, y_train = get_xy_generator_folder(traindata, size)
    X_test, y_test = get_xy_generator_folder(testdata, sizetest)
    X_val, y_val = get_xy_generator_folder(valdata , sizeval)
    return X_train, y_train, X_test, y_test, X_val, y_val
def get_augment_coco():
    print("load augmented data ...")
    """ shall works if we have categories folders inside each augmented folder"""
    image_folder = tfds.ImageFolder("/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented")
    image_folder.info
    trdata = ImageDataGenerator()
    print("change reshape")
    inpaintdata = trdata.flow_from_directory(
        directory="/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Inpainting", target_size=(32, 32))
    tsdata = ImageDataGenerator()
    erasedata = tsdata.flow_from_directory(directory="/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Erasing", target_size=(32,32))
    vldata = ImageDataGenerator()
    noisedata = vldata.flow_from_directory(directory="/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Noise", target_size=(32,32))
    X_train1, y_train1 = next(inpaintdata)
    X_train2, y_train2 = next(erasedata)
    X_train3, y_train3 = next(noisedata)
    size1 = X_train1.shape[0]
    size2 = X_train2.shape[0]
    size3 = X_train3.shape[0]
    X_inp, y_inp = get_xy_generator_folder(inpaintdata, size1)
    X_erase, y_erase = get_xy_generator_folder(erasedata, size2)
    X_noise, y_noise = get_xy_generator_folder(noisedata, size3)
    return X_inp, y_inp, X_noise, y_noise,X_erase, y_erase
def augment_data(Train_X, Train_Y, X, Y):

    test_X = np.concatenate([Train_X, X], axis=0)
    true_test = np.concatenate([Train_Y, Y], axis=0)
    return test_X, true_test
def get_model(model_name):

    path='/home/sondess/sm2672/EU_SESAME/Networks'
    model_path=os.path.join(path, model_name)
    try:
        json_file = open(model_path + '.json', 'r')
        file_content = json_file.read()
        json_file.close()

        model = model_from_json(file_content)
        model.load_weights(model_path + '.h5')

        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        print("exeception")
        model = load_model(model_path + '.h5')

    model.compile(optimizer='adam',loss='categorical_crossentropy', metrics=['accuracy'])

    return model
def load_pkl(filename):
    data= pd.read_pickle(filename)  # , sep='\t',error_bad_lines=False)
    print(" loaded from ", filename)
    return data
def load_knw_data(X_val,y_val, dataset):
    knw_set_preffered = []
    knw_set_top = []
    preffered_y = []

    # aug_XP, aug_YP = augment_data(X_train, y_train, np.array(knw_set_preffered), np.array(preffered_y))
    test_knw_preferred = load_pkl("/home/sondess/sm2672/EU_SESAME/experiments/{}_topgained_testset_knw.sav".format(dataset))
    print("size of test knw preffered", len(test_knw_preferred))
    if len(test_knw_preferred) > 0:
        mylist2 = np.array(test_knw_preferred)
        mylist2 = np.unique(mylist2)
        test_knw_preferred = list(mylist2)

    for input_index in test_knw_preferred:
        input = X_val[input_index]
        label = y_val[input_index]

        knw_set_preffered.append(input)
        preffered_y.append(label)

    return  np.array(knw_set_preffered), np.array(preffered_y)
def parse_arguments():
    text='hardening'
    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments

    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.", choices=['LeNet1','LeNet4','svhn', 'model_cifar10'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN",'coco','leaves'])

    args = parser.parse_args()

    return vars(args)



def repair(dataset):


        # dataset='coco'
        if dataset=='coco':
            categories = ["bear", "bird", 'cat', 'cow', 'dog', 'elephant', 'giraffe', 'horse', 'sheep', 'zebra']
            model_name = "Vgg19_coco_animal_32_32_kera2_9"
            X_train, y_train, X_test, y_test, X_val, y_val,X_val_z, y_val_z=load_coco()
            print("test set size:",X_test.shape[0])
            X_knw,Y_knw= load_knw_data( X_val, y_val, "coco")
            X_inp, y_inp,  X_Noise, y_Noise, X_erase, y_erase= get_augment_coco()
            # X_erase, y_erase,
            print("\n size of synthetic set for retraining \n ",X_inp.shape,y_inp.shape)
            # X_re, Y_re = augment_data(X_inp, y_inp, X_train, y_train)
            # print("\n size of aug_concatenated set for retraining \n ", X_re.shape, Y_re.shape)
            # print("[INFO] evaluating network...\n")
            # loss, acc = model.evaluate(X_test, y_test, verbose=2)
            # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

        elif dataset=='cifar':
            categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            model_name = 'cifar_retrain'
                # 'cifar_original'
            # X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("zeroshot")

            X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("ini")

            X_inp, y_inp, X_erase, y_erase, X_Noise, y_Noise,X_knw_inp, y_knw_inp = load_CIFARVAL('augmented')

            # model = get_model(model_name)
            # model.compile(optimizer='adam',
            #               loss='categorical_crossentropy',
            #               metrics=['accuracy'])

            # train the model
            # history = model.fit(X_train / 255, y_train, epochs=15, batch_size=32)
            # history = model.fit(X_train / 255, to_categorical(y_train), epochs=10, batch_size=32)

            # print("[INFO] evaluating LENET5 network...\n")
            # loss, acc = model.evaluate(X_test, y_test, verbose=2)
            # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))
            # # model.save('/home/sondess/sm2672/EU_SESAME/Networks/cifar_Tuned.h5')
            # # print('model saved')
            # # X_test=X_test[: 5000]
            # # y_test=y_test[: 5000]
            # predictions = model.predict(x=X_test.astype("float32"), batch_size=5)
            # roc = roc_auc_score(y_test, predictions, average="weighted", multi_class="ovr")
            # print("roc_au scores initial\n", roc)

        elif dataset=='leaves':
            categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]
            model_name="grape_leaves_disease"
                # 'leaf_disease_coloured'
            # model_name = 'grape_disease'
            X_train, y_train, X_test, y_test, X_val, y_val, X_test_z,y_test_z= load_leaves_V('original')
            print("test set size:", X_test.shape[0])
            X_inp, y_inp, X_erase, y_erase,X_Noise, y_Noise ,X_knw_inp, y_knw_inp= load_leaves_V('augmented')
            # print("\n size of synthetic set for retraining \n ", X_inp.shape, y_inp.shape)


        # model = get_model('svhn_Tuned')
        model = get_model(model_name)
        loss, acc = model.evaluate(X_test, y_test, batch_size=128, verbose=2)
        # / 255 + cifar
        print('initial test set: {:5.2f}%'.format(100 * acc))
        # train the model
        history = model.fit(X_train/ 255 , y_train, epochs=15, batch_size=32)
        model.save('coco_retrain.h5')
        print('Restored model, accuracy \n')
        loss, acc = model.evaluate(X_test, y_test, batch_size=128, verbose=2)
        # / 255 + cifar
        print('initial test set: {:5.2f}%'.format(100 * acc))

        # Random-Inpainting set
        X_inp_f, X_inp_t, y_inp_f, y_inp_t = train_test_split(X_inp, y_inp, test_size=0.2,
                                                              random_state=1)
        print("X_inp_t", X_inp_t.shape)
        X_aug, Y_aug = augment_data(X_inp_f[:10], y_inp_f[:10], X_train, y_train)
        X_test_inp, y_test_inp = augment_data(X_inp_t[:3], y_inp_t[:3], X_test, y_test)

        # deepknw guided inpainting
        X_knw_f, X_knw_t, y_knw_f, y_knw_t = train_test_split(X_knw_inp, y_knw_inp, test_size=0.2,
                                                              random_state=1)
        X_kp, Y_kp = augment_data(X_knw_f, y_knw_f, X_train, y_train)
        X_test_knw_inp, y_test_knw_inp = augment_data(X_knw_t, y_knw_t, X_test, y_test)


        # Erase set
        X_e_f, X_e_t, y_e_f, y_e_t = train_test_split(X_erase, y_erase, test_size=0.2,
                                                      random_state=1)
        print("X_e_t",X_e_t.shape)
        X_e, Y_e = augment_data(X_e_f[:10], y_e_f[:10], X_train, y_train)
        X_test_Erase, y_test_Erase = augment_data(X_e_t[:3], y_e_t[:3], X_test, y_test)






        # # deepknw  set
        X_k, Y_k = load_knw_data(X_val, y_val, "coco")
        X_k_f, X_k_t, y_k_f, y_k_t = train_test_split(X_k, Y_k, test_size=0.2,
                                                      random_state=1)
        X_kn, Y_kn = augment_data(X_k_f[:10], y_k_f[:10], X_train, y_train)
        X_test_kn, y_test_kn = augment_data(X_k_t[:3], y_k_t[:3], X_test, y_test)
        #
        # # noise set
        X_n_f, X_n_t, y_n_f, y_n_t = train_test_split(X_Noise, y_Noise, test_size=0.2, random_state=1)
        X_no, Y_no = augment_data(X_n_f[:10], y_n_f[:10], X_train, y_train)
        X_test_noise, y_test_noise = augment_data(X_n_t[:3], y_n_t[:3], X_test, y_test)
        #
        # # random set
        print("X_val",X_val.shape)
        X_v_f, X_v_t, y_v_f, y_v_t = train_test_split(X_val[10:20], y_val[10:20], test_size=0.2,
                                                      random_state=1)
        X_random, Y_random = augment_data(X_v_f[:10], y_v_f[:10], X_train, y_train)
        X_test_random, y_test_random = augment_data(X_v_t[:3], y_v_t[:3], X_test, y_test)

        print("Initial train set  ", X_train.shape[0])
        # print("train Random-guided inpainting  ", X_inp_f.shape[0])
        # print("train knw-guided inpainting  ", X_knw_f.shape[0])
        # print("train Erase  ", X_e_f.shape[0])
        print("train Random  ", X_v_f.shape[0])
        print("train knw set", X_k.shape[0])
        print("train noise  ", X_n_f.shape[0])


        #Retrain model with Random Inpainting
        model_tune1=model
        print("\n Augmented\n ", X_aug.shape, Y_aug.shape)
        H = model_tune1.fit(X_aug, Y_aug, epochs=15, batch_size=32)
        # model_tune1.save('home/sondess/sm2672/EU_SESAME/Networks/'+model_name + 'Knw-tuned-inpaint_{}.h5'.format(dataset))
        print("#############################################")
        print('Random inpainting model, accuracy \n')
        loss, acc1 = model_tune1.evaluate(X_test, y_test, batch_size=128, verbose=2)
        print('accuracy on initial set: {:5.2f}%'.format(100 * acc1))

        loss, acc2 = model_tune1.evaluate(X_inp_t[:50], y_inp_t[:50], batch_size=128, verbose=2)
        print('accuracy on only augmented test set: {:5.2f}%'.format(100 * acc2))

        loss, acc3 = model_tune1.evaluate(X_test_inp, y_test_inp, batch_size=128, verbose=2)
        print('accuracy on infected test set: {:5.2f}%'.format(100 * acc3))

        ### Retrain model with Knw guided inpainting
        model_knw_inp = model
        H = model_knw_inp.fit(X_kp/ 255 , Y_kp, epochs=15, batch_size=32)
        model_knw_inp.save(
            'home/sondess/sm2672/EU_SESAME/Networks/' + model_name + 'fine-tuned_knw_guided_inp_{}.h5'.format(dataset))


        modelerase = model
        H = modelerase.fit(X_e, Y_e, epochs=15, batch_size=32)
        modelerase.save(
            'home/sondess/sm2672/EU_SESAME/Networks/' + model_name + 'fine-tuned-erase_{}.h5'.format(dataset))

        modelrandom = model
        H = modelrandom.fit(X_random, Y_random, epochs=15, batch_size=32)
        modelrandom.save(
            'home/sondess/sm2672/EU_SESAME/Networks/' + model_name + 'fine-tuned-noise_{}.h5'.format(dataset))
        #
        modelknw = model
        H = modelknw.fit(X_kn, Y_kn, epochs=15, batch_size=32)
        modelknw.save('home/sondess/sm2672/EU_SESAME/Networks/' + model_name + 'fine-tuned-noise_{}.h5'.format(dataset))
        # #
        modelnoise = model
        H = modelnoise.fit(X_no, Y_no, epochs=15, batch_size=32)
        modelnoise.save('home/sondess/sm2672/EU_SESAME/Networks/' + model_name + 'fine-tuned-noise.h5')

        # Evaluation results:
        # #####################
        #######################
        print("#############################################")
        print("########## EVALUATION ##########################")
        print("original test set ", X_test.shape)
        print('Restored model, accuracy \n')
        loss, acc = model.evaluate(X_test , y_test, batch_size=128, verbose=2)
        # / 255 + cifar
        print('initial test set: {:5.2f}%'.format(100 * acc))
        loss, acc = model.evaluate(X_test_random , y_test_random, verbose=2)
        print('Random +original, accuracy: {:5.2f}%'.format(100 * acc))

        # loss, acc = model.evaluate(X_test_knw_inp/ 255 , y_test_knw_inp, batch_size=128, verbose=2)
        print('random inpainting + original test: {:5.2f}%'.format(100 * acc))
        loss, acc = model.evaluate(X_inp_t , y_inp_t, batch_size=128, verbose=2)
        print('random inpainting alone: {:5.2f}%'.format(100 * acc))

        # loss, acc = model.evaluate(X_test_knw_inp/ 255 , y_test_knw_inp, batch_size=128, verbose=2)
        # print('knw inpainting + original test: {:5.2f}%'.format(100 * acc))
        # loss, acc = model.evaluate(X_knw_t/ 255 , y_knw_t, batch_size=128, verbose=2)
        # print('knw inpainting alone: {:5.2f}%'.format(100 * acc))


        loss, acc = model.evaluate(X_test_Erase, y_test_Erase, batch_size=128, verbose=2)
        print('Erase + original test: {:5.2f}%'.format(100 * acc))
        loss, acc = model.evaluate(X_e_t , y_e_t, batch_size=128, verbose=2)
        print('Erase alone: {:5.2f}%'.format(100 * acc))

        loss, acc = model.evaluate(X_test_kn , y_test_kn, batch_size=128, verbose=2)
        print('knw + original test: {:5.2f}%'.format(100 * acc))
        loss, acc = model.evaluate(X_k_t , y_k_t, batch_size=128, verbose=2)
        print('knw alone: {:5.2f}%'.format(100 * acc))

        loss, acc = model.evaluate(X_test_noise , y_test_noise, batch_size=128, verbose=2)
        print('Noise + original test: {:5.2f}%'.format(100 * acc))
        loss, acc = model.evaluate(X_n_t , y_n_t, batch_size=128, verbose=2)
        print('Noise alone: {:5.2f}%'.format(100 * acc))



        print("#############################################")
        print('Knw-guided inpainting model, accuracy \n')

        loss, acc = model_knw_inp.evaluate(X_test/ 255, y_test, verbose=2)
        print('accuracy on initial set: {:5.2f}%'.format(100 * acc))

        loss, acc = model_knw_inp.evaluate(X_knw_t/ 255, y_knw_t, batch_size=128, verbose=2)
        print('accuracy on only augmented test set: {:5.2f}%'.format(100 * acc))

        loss, acc = model_knw_inp.evaluate(X_test_knw_inp/ 255 , y_test_knw_inp, verbose=2)
        print('accuracy on infected test set: {:5.2f}%'.format(100 * acc))


        print("#############################################")
        print('Erase model, accuracy \n')

        loss, acc = modelerase.evaluate(X_test, y_test, batch_size=128, verbose=2)
        print('original test: {:5.2f}%'.format(100 * acc))
        loss, acc = modelerase.evaluate(X_e_t, y_e_t, batch_size=128, verbose=2)
        print('Erase alone: {:5.2f}%'.format(100 * acc))
        loss, acc = modelerase.evaluate(X_test_Erase, y_test_Erase, verbose=2)
        print('Erase + original test accuracy: {:5.2f}%'.format(100 * acc))

        print("#############################################")
        print('Random model, accuracy \n')

        loss, acc = modelrandom.evaluate(X_test, y_test, verbose=2)
        print('Random on initial test set accuracy: {:5.2f}%'.format(100 * acc))
        loss, acc = modelrandom.evaluate(X_v_t, y_v_t, batch_size=128, verbose=2)
        print('Random alone: {:5.2f}%'.format(100 * acc))
        loss, acc = modelrandom.evaluate(X_test_random, y_test_random, verbose=2)
        print('Random + original test accuracy: {:5.2f}%'.format(100 * acc))

        print("#############################################")
        # print('knw model, accuracy \n')
        loss, acc = modelknw.evaluate(X_test, y_test, batch_size=128, verbose=2)
        print('original test: {:5.2f}%'.format(100 * acc))
        loss, acc = modelknw.evaluate(X_k_t, y_k_t, batch_size=128, verbose=2)
        print('knw alone: {:5.2f}%'.format(100 * acc))
        loss, acc = modelknw.evaluate(X_test_kn, y_test_kn, verbose=2)
        print('KNW+ original, accuracy: {:5.2f}%'.format(100 * acc))

        print("#############################################")
        print('Noise model, accuracy \n')
        loss, acc = modelnoise.evaluate(X_test, y_test, verbose=2)
        print('accuracy on initial set: {:5.2f}%'.format(100 * acc))

        loss, acc = modelnoise.evaluate(X_n_t, y_n_t, batch_size=128, verbose=2)
        print('accuracy on only augmented  noise test set: {:5.2f}%'.format(100 * acc))

        loss, acc = modelnoise.evaluate(X_test_noise, y_test_noise, verbose=2)
        print('NOISE+ orig, accuracy: {:5.2f}%'.format(100 * acc))


        # with open('./results/' + file_nm, 'w') as out_file:
        #     tsv_writer = csv.writer(out_file, delimiter='\t')
        #     tsv_writer.writerow(entete)
        #     for l in rsl:
        #         tsv_writer.writerow(l)


        #

        # predictions = model_tune.predict(x=X_test.astype("float32"), batch_size=BS)
        # print(classification_report(y_test.argmax(axis=1),
        #                             predictions.argmax(axis=1), target_names=categories))
        # roc=roc_auc_score(y_test, predictions, average = "weighted", multi_class = "ovr" )
        # print("roc_au scores inpainting\n",roc)

        # accuracy = zero_one_loss(y_test, np.array(predictions))
        # error_rate = 1 - accuracy
        # # # plot the training loss and accuracy
        # N = np.arange(0, BS)
        # plt.style.use("ggplot")
        # plt.figure()
        # # plt.plot(N, H.history["loss"], label="train_loss")
        # # plt.plot(N, H.history["val_loss"], label="val_loss")
        # plt.plot(N, H.history["accuracy"], label="train_acc")
        # plt.plot(N, H.history["val_accuracy"], label="val_acc")
        # plt.title("Vgg19 re-Training Loss and Accuracy on COCO -animal Dataset")
        # plt.xlabel("Epoch #")
        # plt.ylabel("Loss/Accuracy")
        # plt.legend(loc="lower left")
        # plt.savefig("plot_generated_dataset.png")
        return 0
def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """



    text = 'Repairing for EDDIS Deep Learning Componenets'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments

    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.", choices=['LeNet5','Vgg19','AllConvNet', 'model_cifar10'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (coco, Leaves or cifar10).", choices=["coco","cifar10","Leaves"])
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                            to measure coverage", choices=['knw', 'idc'])
    #
    parser.add_argument("-method", "--method", help="1 for repairing", type=int)
    parser.add_argument("-repair", "--repair", help="the continous learning paradigm", choices=['CL','CLClass','ClTask'])
    parser.add_argument("-it", "--iteration", help=" number of iteration for data augmentation within the fuzzing process", choices=[2,5, 10],
                        type=int)

    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)


    parser.add_argument("-LOG", "--logfile", help="path to log file")

    args = parser.parse_args()

    return vars(args)
if __name__ == "__main__":
    args = parse_arguments()
    model = args['model'] if args['model'] else 'AllConvNet'
    dataset = args['dataset'] if args['dataset'] else 'coco'
    approach = args['approach'] if args['approach'] else 'knw'
    method = args['method'] if args['method'] else 1
    repair = args['repair'] if args['repair'] else 'CL'
    iteration = args['it'] if args['it'] else 2
    logfile_name = args['logfile'] if args['logfile'] else 'resultknw.log'
    logfile = open(logfile_name, 'a')

    repair(model,dataset,repair,iteration,method, approach)
