import argparse
# from absl import app
import tensorflow_datasets as tfds

# Helper libraries
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import os
from datetime import datetime
from tensorflow.keras.models import model_from_json, load_model, save_model
# import multiprocessing
# import itertools
# from Dataprocessing import load_MNISTVAL, load_CIFARVAL, load_MNIST, load_CIFAR, load_SVHN_
# from utils import filter_val_set, get_trainable_layers
# from utils import generate_adversarial, filter_correct_classifications
# from Coverages.idc import ImportanceDrivenCoverage
# from Coverages.neuron_cov import NeuronCoverage
# from Coverages.tkn import DeepGaugeLayerLevelCoverage
# from Coverages.kmn import DeepGaugePercentCoverage
# from Coverages.ss import SSCover
# from Coverages.sa import SurpriseAdequacy
# from Coverages.knw import KnowledgeCoverage
from Coverages.TrKnw import *
from tensorflow.keras import applications
from tensorflow.python.client import device_lib
import tensorflow
import os

os.environ['TF_GPU_ALLOCATOR']="cuda_malloc_async"
os.environ["TF_CPP_VMODULE"]="gpu_process_state=10,gpu_cudamallocasync_allocator=10"
__version__ = 1.3
#
print(keras.__version__)
print(tensorflow.__version__)


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """



    text = 'Knowledge Coverage for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments
    parser.add_argument("-U", "--adv_use",  default=False, type=bool, help="use adversarial attacks")
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.", choices=['LeNet1','LeNet4','svhn', 'model_cifar10'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN"])
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                            to measure coverage", choices=['knw', 'idc'])
    # parser.add_argument("-P", "--percentage", help="the percentage of TrKnw neurons to be deployed", type=float)
    parser.add_argument("-K", "--nbr_Trknw", help="the number of TrKnw neurons to be deployed", type=float)
    parser.add_argument("-HD", "--HD_thre", help="a threshold value used\
                            to identify the type of TrKnw neurons.", type=float)
    parser.add_argument("-Tr", "--TrKnw", help="Type of selected TrKnw neurons based on HD values range.", choices=['top', 'least', 'preferred'])
    parser.add_argument("-Sp", "--split", help="percentage of test data to be tested", type=float)
    parser.add_argument("-ADV", "--adv", help="name of adversarial attack", choices=['mim', 'bim', 'fgsm', 'pgd'])

    parser.add_argument("-C", "--class", help="the selected class", type=int)

    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)


    parser.add_argument("-LOG", "--logfile", help="path to log file")

    args = parser.parse_args()

    return vars(args)
def Fuzzing(approach,modelpath,dataset,TypeTrknw, percent,selected_class,threshold, attack,split,use_augment,augment):
    model_path = 'Networks/'+modelpath

    img_rows, img_cols, img_channel = 32, 32, 3

    model_name = model_path.split('/')[-1]
    print(model_name)
    if model_name == 'grape':
        if tf.executing_eagerly():
            tf.compat.v1.disable_eager_execution()
        model = tf.keras.models.load_model("./Networks/leaf_disease_coloured.h5")

        print("Model grape disease detection is loaded")
    elif model_name == 'vgg16':
        model = applications.VGG16(weights='imagenet', include_top=False,
                                   input_shape=(img_rows, img_cols, img_channel))
        print("Model VGG 16 is loaded")

    elif model_name == 'ImagNet':
        model =tf.keras.applications.MobileNetV2(include_top=True,
                                          weights='imagenet')
        model.trainable = False
        model.compile(optimizer='adam',
                                 loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                                 metrics=['accuracy'])

        print("Model imagenet is loaded")
    else:

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
            model = load_model(model_path + '.hdf5')


    trainable_layers = get_trainable_layers(model)
    dense_layers=get_dense_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'
    isExist = os.path.exists(experiment_folder)
    if not isExist:
        os.makedirs(experiment_folder)
    dataset_folder = 'dataset'
    isExist = os.path.exists(dataset_folder)
    if not isExist:
        os.makedirs(dataset_folder)



    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = []  # SKIP LAYERS FOR NC, KMNC, NBC
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print("Skipping layers:", skip_layers)
    ####################
    # print("the model layers")
    for idx,lyr in enumerate(model.layers):
        print(idx,lyr.name)


    if approach == 'knw':
        model_folder = 'Networks'
        method = 'idc'

        knw = KnowledgeCoverage(model, dataset, model_name, subject_layer, trainable_layers,dense_layers, method, percent, threshold,attack, skip_layers, nbr_Trknw,use_augment,augment,selected_class=1)


        Knw_coverage, covered_TrKnw, combinations, max_comb, testsize, zero_size,Trkneurons= knw.run(split,TypeTrknw,use_adv)

        print("The model Transfer Knowledge Neurons number: ", covered_TrKnw)
        # print("type",TypeTrknw)
        print("The percentage of the used neurons out of all Transfer Knowledge Neurons : ",percent)
        if use_adv:
            print("Deployed Adversarials attacks", attack)
        if split>0:
            print("Test set is splited and only %.2f%% is used" %(1-split))
        print("The test set coverage: %.2f%% for dataset  %s " % (Knw_coverage, dataset))

        print("Covered combinations: ", len(combinations))
        print("Total combinations:", max_comb)
        line=[model_name, dataset, testsize, zero_size, Knw_coverage, covered_TrKnw, TypeTrknw, (1-split),len(combinations),
                     max_comb,Trkneurons, attack]





    else:
        print("other method")

    logfile.close()
    return line


if __name__ == "__main__":

                args = parse_arguments()
                method= args['method'] if args['method'] else 1
                fuzzer = args['fuzzer'] if args['fuzzer'] else "RInp"
                iteration = args['it'] if args['it'] else 10
                repair = args['repair'] if args['repair'] else False
                model = args['model'] if args['model'] else 'AllConvNet'
                dataset = args['dataset'] if args['dataset'] else 'Leaves'
                approach = args['app'] if args['app'] else 'knw'
                layer = args['layer'] if not args['layer'] == None else -1
                logfile_name = args['logfile'] if args['logfile'] else 'fuzz.log'
                logfile = open(logfile_name, 'a')
                startTime = time.time()

                Fuzzing(method, fuzzer, iteration, approach, model, dataset, approach, repair, layer)

                logfile.close()
                endTime = time.time()
                elapsedTime = endTime - startTime
                print("Elapsed Time = %s" % elapsedTime)







