import tensorflow as tf
import os
from utils import *

import pandas as pd
import numpy as np

from keras.models import model_from_json, load_model, save_model

from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from Coverages.knw import *
'''

Each layer has two weight matrices - first one for link weights, 
and the second one for bias weights of every neuron of that layer.
'''


'''

For weight pruning, we will create a dictionary of the weights, with every key being a three valued tuple. 
The first value having the layer number, second and third values having the row and column number from that layer's weight matrix.

Each row number is in fact the neuron number from previous layer, and each column number is the neuron number of that particular layer.
 So, for neuron pruning, unlike weight pruining where we want to create a weights dictionary, 
 here we will have a dictionary that contains column arrays. 
The key will be a two valued tuple. First value representing the layer number, and the second one representing the column number.
'''


'''
We will sort the weights in the weights dictionary according to their absolute values, 
and the weight vectors in the neuron dictionary according to their L2 norm.

We will then map the weights (or columns) from the sorted dictionaries on to the trained neural network
and set the weights (or columns) to zero to obtain our compresed Neural Networks (with the set pruing level).
'''

#Pruning percentages

K = [0, 25, 50, 60, 70, 80, 90, 95, 97, 99]


def weight_pruning(model,K):
    trained_model = load_model(model)

    trained_model.layers
    total_no_layers = len(trained_model.layers)
    print(total_no_layers)
    all_weights = {}

    for layer_no in range(total_no_layers - 1):         #All except the final layer                                                                          #only the first four dense layers are to be pruned
        layer_weights = (pd.DataFrame(trained_model.layers[layer_no].get_weights()[0]).stack()).to_dict()
        layer_weights = { (layer_no, k[0], k[1]): v for k, v in layer_weights.items() }
        all_weights.update(layer_weights)
    all_weights_sorted = {k: v for k, v in sorted(all_weights.items(), key=lambda item: abs(item[1]))}
    total_no_weights = len(all_weights_sorted)
    print(total_no_weights)
    weight_pruning_scores = []

    for pruning_percent in K:

        new_model = load_model(model)
        new_weights = trained_model.get_weights().copy()

        prune_fraction = pruning_percent / 100
        number_of_weights_to_be_pruned = int(prune_fraction * total_no_weights)
        weights_to_be_pruned = {k: all_weights_sorted[k] for k in
                                list(all_weights_sorted)[:  number_of_weights_to_be_pruned]}

    for k, v in weights_to_be_pruned.items():
        new_weights[k[0]][k[1], k[2]] = 0

    for layer_no in range(total_no_layers - 1):
        new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0],
                                                          new_weights[layer_no].shape[1])
        new_model.layers[layer_no].set_weights(new_layer_weights)
            #     to evaluate
            # new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            # weight_pruning_scores.append(new_score[1])
    return new_model



def Neuron_Pruning(model, neuron_to_prune):
    try:
        json_file = open(model + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        trained_model = model_from_json(file_content)
        trained_model.load_weights(model_path + '.h5')

        # Compile the model before using
        trained_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        trained_model = load_model(model + '.hdf5')
        # trained_model = load_model(model + '.hdf5')
    neuron_pruning_scores = []
    # print("the model layer\n",trained_model.layers)
    trainable_layers = get_trainable_layers(trained_model)
    denses=get_dense_layers(trained_model)
    non_trainable_layers = list(set(range(len(trained_model.layers))) - set(trainable_layers))
    # print('Trainable layers: ', trainable_layers)
    print('Non trainable layers: ' + str(non_trainable_layers))
    print("dense layers:", denses)

    # total_no_layers = len(trained_model.layers)
    total_no_layers = len(trainable_layers)
    print("number of layer:", total_no_layers)
    all_neurons = {}
    all_neurons_new = {}
    neurons_to_be_pruned = []
    neurons_prun={}
    for layer_no in (trainable_layers[:-1]):
        print("layer_number",layer_no)
        if layer_no in denses:
            # layer_n_prune={}
            # layer_neurons = {}
            # layer_neurons_df = pd.DataFrame(trained_model.layers[layer_no].get_weights()[0])
            #
            # for i in range(len(layer_neurons_df.columns)):
            #     layer_neurons.update({i: np.array(layer_neurons_df.iloc[:, i])})
            # layer_neurons = {(layer_no, k): v for k, v in layer_neurons.items()}
            #
            # all_neurons.append(layer_neurons)
            # all_neurons_new.update(layer_neurons)

            layer_neurons = {}
            layer_neurons_df = pd.DataFrame(trained_model.layers[layer_no].get_weights()[0])

            for i in range(len(layer_neurons_df.columns)):
                layer_neurons.update({i: np.array(layer_neurons_df.iloc[:, i])})

            layer_neurons = {(layer_no, k): v for k, v in layer_neurons.items()}
            all_neurons.update(layer_neurons)
    all_neurons_sorted = {k: v for k, v in
                          sorted(all_neurons.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
    total_no_neurons = len(all_neurons_sorted)
    total_no_neurons
    # for n in neuron_to_prune.keys():
    #         print(n, '->', neuron_to_prune[n])
    #         for l in all_neurons:
    #
    #             for k,v in l.items():
    #                  if n==k[1] and neuron_to_prune[n]==k[0] :
    #
    #                     print("the neuron,", n)
    #                     neurons_to_be_pruned.append(n)
    #                     layer_n_prune.update({(layer_no, k[1]): v})
    #
    #
    #
    #                     neurons_prun.update(layer_n_prune)
    # all_neurons_sorted = {k: v for k, v in
    #                        sorted(all_neurons_new.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
    #
    # total_no_neurons = len(neurons_prun)
    # print(total_no_neurons)


    # for pruning_percent in K:

    new_model = load_model(model+ '.hdf5')
    new_weights = trained_model.get_weights().copy()

    prune_fraction = 100 / 100
    number_of_neurons_to_be_pruned = int(prune_fraction * total_no_neurons)
    neurons_to_be_pruned = {k: all_neurons_sorted[k] for k in
                            list(all_neurons_sorted)[: number_of_neurons_to_be_pruned]}

    for k, v in neurons_to_be_pruned.items():

        if k[0] in denses:
            print("again",k[0])
            print(k[1])
            new_weights[k[0]][:, k[1]] = 0

    for layer_no in range(total_no_layers - 1):
        new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0],
                                                          new_weights[layer_no].shape[1])
        new_model.layers[layer_no].set_weights(new_layer_weights)


    return new_model





if __name__ == "__main__":
        dataset = 'svhn'
        path='experiments/'
        X_train, y_train, X_test, Y_test, X_val, Y_val = load_SVHN_("ini")
        neuron_pruning_scores={}
        model_path='Networks/svhn'
        model_name=model_path.split('/')[-1]

        transfer_neurons=load_KnwTrNeurons(path+'svhn_0.04_svhn_knw.pkl')
        print(transfer_neurons)
        new_model=Neuron_Pruning(model_path, transfer_neurons)

        new_score = new_model.evaluate(X_test, Y_test, verbose=0)
        neuron_pruning_scores.update({model_name:new_score[1]})


