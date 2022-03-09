import tensorflow as tf
import os
from utils import *

import pandas as pd
import numpy as np

from keras.models import model_from_json, load_model, save_model
import tensorflow as tf
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
# from kerassurgeon import Surgeon, identify
# from kerassurgeon.operations import delete_channels, delete_layer
import os
import numpy as np
import math
# from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.datasets import fashion_mnist
# from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Activation
# from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from Coverages.knw import *
from keras.optimizer_v2 import adam
# from kerassurgeon.operations import delete_layer, insert_layer, delete_channels
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





def get_layer_weights(model, layer=None):
    """function to return weights array for one or all conv layers of a Keras model"""
    if layer or layer == 0:
        weight_array = model.layers[layer].get_weights()[0]

    else:
        weights = [model.layers[layer_ix].get_weights()[0] for layer_ix in range(len(model.layers)) \
                   if 'conv' in model.layers[layer_ix].name]
        weight_array = [np.array(i) for i in weights]

    return weight_array




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
        new_model.layers[layer_no].trainable = True
        new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0],
                                                          new_weights[layer_no].shape[1])
        new_model.layers[layer_no].set_weights(new_layer_weights)
            #     to evaluate
            # new_score = new_model.evaluate(X_test_norm, Y_test_onehot, verbose=0)
            # weight_pruning_scores.append(new_score[1])
    return new_model



def Neuron_Pruning(model_path, neuron_to_prune):
    model_name = model_path.split('/')[-1]
    try:
        json_file = open(model_path  + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
        file_content = json_file.read()
        json_file.close()

        trained_model = model_from_json(file_content)
        trained_model.load_weights(model_path + '.h5')

        # Compile the model before using
        trained_model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    except:
        trained_model = load_model(model_path + '.hdf5')

    neuron_pruning_scores = []
    # print("the model layer\n",trained_model.layers)
    trainable_layers = get_trainable_layers(trained_model)
    dense_layers=get_dense_layers(trained_model)
    non_trainable_layers = list(set(range(len(trained_model.layers))) - set(trainable_layers))
    print('Trainable layers: ', trainable_layers)
    print('Non trainable layers: ' + str(non_trainable_layers))

    total_no_layers = len(trained_model.layers)
    print("number of layer:", total_no_layers)

    neurons_to_be_pruned = []
    all_neurons= {}
    layer_n_prune={}
    for layer_no in range (len(trained_model.layers)):

        pruned_layer = trained_model.layers[layer_no]
        if "dense" in pruned_layer.name:
            layer_neurons_df = pd.DataFrame(get_layer_weights(trained_model, layer_no))
            print(layer_no)
            layer_neurons = {}

            for i in range(len(layer_neurons_df.columns)):
                layer_neurons.update({i: np.array(layer_neurons_df.iloc[:, i])})
            layer_neurons = {(layer_no, k): v for k, v in layer_neurons.items()}
            all_neurons.update(layer_neurons)
        else:
            weights_array=get_layer_weights(trained_model)


    neurons_to_be_pruned = {k: v for k, v in all_neurons.items() if k[1] in neuron_to_prune.keys()}
    print("neurons to prune", len(neurons_to_be_pruned))

    all_neurons_sorted = {k: v for k, v in
                              sorted(all_neurons.items(), key=lambda item: np.linalg.norm(item[1], ord=2, axis=0))}
    total_no_neurons = len(all_neurons_sorted)
    print("tot",total_no_neurons)



    new_model = load_model(model_path + '.hdf5')


    for k, v in neurons_to_be_pruned.items():
        pruned_layer = trained_model.layers[k[0]]
        pruned_layer.trainable = True
        layer_no=k[0]
        new_weight=get_layer_weights(trained_model, layer_no)
        new_weight[:, k[1]] = 0
        _bias = trained_model.layers[layer_no].get_weights()[1]


        new_model.layers[layer_no].set_weights([new_weight,_bias])
        # print(new_layer_weights.shape)
        # new_model.layers[layer_no].set_weights(np.array(new_weight))


    # for layer_no in trained_model.layers:
    #
    #         new_layer_weights = new_weights[layer_no].reshape(1, new_weights[layer_no].shape[0],
    #                                                                   new_weights[layer_no].shape[1])
    #         new_model.layers[layer_no].set_weights(new_layer_weights)
    return new_model,len(neurons_to_be_pruned)



def get_layer(knw_neurons):

    res = defaultdict(list)
    for key, val in sorted(knw_neurons.items()):
            res[val[0]].append(key)
    return res

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
if __name__ == "__main__":
        dataset = 'svhn'
        files=['svhn_0.1_svhn_knw.pkl','svhn_0.04_svhn_knw.pkl','svhn_0.05_svhn_knw.pkl']
        path='experiments/'
        X_train, y_train, X_test, Y_test, X_val, Y_val = load_SVHN_("ini")
        neuron_pruning_scores={}
        model_path='Networks/svhn'
        model_name=model_path.split('/')[-1]
        trained_model = load_model(model_path + '.hdf5')
        initial_score = trained_model.evaluate(X_test, Y_test, verbose=0)
        print(initial_score)
        neuron_pruning_scores.update({model_name:initial_score[1]})
        scores=[]
        for filename in files:
            transfer_neurons=load_KnwTrNeurons(path+filename)
            print("transfer_neurons\n",transfer_neurons)
            # new_model=Neuron_Pruning(model_path, transfer_neurons)
            pruned_model,nbrpruned=Neuron_Pruning(model_path, transfer_neurons)
            opt = adam.Adam(0.0001)
            pruned_model.compile(loss='categorical_crossentropy',
                             optimizer=opt,
                             metrics=['accuracy'])


            new_score = pruned_model.evaluate(X_val, Y_val, verbose=0)
            print(new_score)
            neuron_pruning_scores[nbrpruned]=new_score[1]
            scores.append(new_score[1])
        print(neuron_pruning_scores)
        print(scores)
        plt.figure(figsize=(8, 4))
        plt.plot(pd.DataFrame(weight_pruning_scores).set_index(pd.Series(K), drop=True), color='r')
        plt.plot(pd.DataFrame(neuron_pruning_scores).set_index(pd.Series(K), drop=True), color='b')
        plt.title('Effect of Pruning on accuracy', weight='bold', fontsize=16)
        plt.ylabel('Score', weight='bold', fontsize=14)
        plt.xlabel('Pruning Percentage (K)', weight='bold', fontsize=14)
        plt.xticks(weight='bold', fontsize=12)
        plt.yticks(weight='bold', fontsize=12)
        plt.legend(['Weight Pruning', 'Neuron Pruning'], loc='best', prop={'size': 14})
        plt.grid(color='y', linewidth='0.5')
        plt.show()



