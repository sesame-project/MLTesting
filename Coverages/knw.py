
import csv
import numpy as np
from utils import *
from leaves_data_processing import load_grape_leaves
from Dataprocessing import load_leaves
import statistics
import operator
import pandas as pd
from statistics import mean
pd.options.mode.chained_assignment = None
import os
import pickle
from collections import defaultdict
from Coverages.idc_knw import *
from sklearn import externals
import joblib
from Dataprocessing import load_MNISTVAL, load_CIFARVAL, load_driving_data, load_SVHN_, load_EMNIST, load_MNIST, \
    load_CIFAR, load_MNIST_32, split_data, load_Imagnet, load_cifar_vgg, load_coco, load_coco_augmented
from hellinger_distance import *
from lrp_toolbox.model_io import write, read


def scaleHD(points, val, rmax=1, rmin=0):
    maxval = points[max(points, key=points.get)]

    minval = points[min(points, key=points.get)]

    X_std = (float(val) - minval) / (maxval - minval)
    # X_scaled = X_std * (rmax - rmin) + rmin
    # print("the scaled value", val)
    return X_std


def get_adv(attack, dataset, X_test, Y_test):
    adv_image_all = []

    adv_image_all.append(np.load('dataset/adv_image/{}_{}_image.npy'.format(attack, dataset)))

    adv_image_all = np.concatenate(adv_image_all, axis=0)
 
    test = np.concatenate([X_test, adv_image_all], axis=0)

    return test

def get_stat(preferred_neurons,type):
    maxvalpref = preferred_neurons[max(preferred_neurons, key=preferred_neurons.get)]
    minvalpref = preferred_neurons[min(preferred_neurons, key=preferred_neurons.get)]
    median_H_D = np.median(list(preferred_neurons.values()))

    return 0
def save_KnwTrNeurons(neurons, filename):
    
    filename = filename + '_knw.sav'
    joblib.dump(neurons, filename)
    
    return


def load_KnwTrNeurons(filename):
   
    neurons =joblib.load(filename, mmap_mode='r')
    return neurons


def filter_neurons_by_layer(TopN, skip_layers):

    relevant = {}
    layers = []

    for x in TopN:
        if x[0] in relevant:
            relevant[x[0]].append(x[1])
        else:
            relevant[x[0]] = [x[1]]

    return relevant


class KnowledgeCoverage:

    def __init__(self, model, data_set, model_name, subject_layer, trainable_layers, dense_layers, method, percent,thershold, attack, skip_layers, nbr_Trknw,  use_augment,augment,scaler=default_scale,selected_class=1):

        self.activation_table = defaultdict(bool)
        self.dense_layers = dense_layers
        self.model = model
        self.attack = attack
        self.scaler = scaler
        self.data_set = data_set
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.model_name = model_name
        self.subject_layer = subject_layer
        self.trainable_layers = trainable_layers
        self.selected_class = selected_class
        self.method = method
        self.percentage = percent
        self.distance_thershold = thershold
        self.user_input = nbr_Trknw
        self.use_augment=use_augment
        self.augment=augment

    def KnwTransfer(self, test_inputs, statep):
        """
               :param test_inputs: Inputs

               """
        saved_dir = "data/" + statep + '_' + self.data_set + '_layers.sav'
        fields_to_generate = ['inputs', 'activations', 'max_activations', 'max_activation_index', 'neuron_index']
        data = []

        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        
        filter_count_by_layer = {}
        layer_count = len(outs)
        neuron_dictionary = {}

        for input_index in range(len(test_inputs)):  # out_for_input is output of layer for single input
            neuron_layer = {}
            # for layer_index in range(layer_count):  # layer_out is output of layer for all inputs
            for layer_index, layer_out in enumerate(outs):

                
                    inside_layer = {}
            
                    out_for_input = outs[layer_index][input_index]

                    filter_outs = np.zeros((out_for_input.shape[-1],))
                    filter_count_by_layer[layer_index] = len(filter_outs)
                    max_act = np.max(out_for_input)  # maximum activation given by an input for one layer
                    
                    neuron_id = np.argmax(out_for_input, axis=None)  # ==> this gives over 1k of neurons
                    index_neuron = np.unravel_index(np.argmax(out_for_input), out_for_input.shape)
               
                    for filter_index in range(out_for_input.shape[-1]):
                        maxact = np.max(out_for_input[..., filter_index])
                        neuronTuple = {(layer_index, filter_index): maxact}
                        inside_layer.update(neuronTuple)

                    filter_key = max(inside_layer, key=inside_layer.get)
                   
                    val_max = inside_layer.get(filter_key)
                   
                    neuron_dictionary.update({layer_index: filter_key})
                    global_neuron_index = (layer_index, (neuron_id, index_neuron))
                    
                    data.append([input_index, layer_index, max_act, global_neuron_index, neuron_id])
                    

        df = pd.DataFrame(data,
                          columns=fields_to_generate)
       
        with open(saved_dir, 'wb') as out_file:
            pickle.dump(df, out_file)

       
        return data

    def coverage_score(self, topN, testdata, train_inputs, test_inputs, subsetTop):

        covered_neurons = []
        testmodel = testdata

        method = self.method

        modeltest_neurons = list(testmodel['max_activation_index'].unique())
        Trknw_shared_neurons = find_shared_neurons(modeltest_neurons, topN)
        number_Trknw_shared_neurons = len(Trknw_shared_neurons)
        test_features = {}
        for neuron in Trknw_shared_neurons:
            model_data = testmodel[testmodel['max_activation_index'] == neuron]
            model_data["normalized_max_activations"] = model_data["max_activations"].apply(
                lambda x: x / model_data["max_activations"].sum())

            
            model_dict = {}
            unique_features = model_data['inputs'].unique()
            test_features[neuron] = len(list(unique_features))

        if method == 'feature_l':
            for n in topN:
                    if n in test_features:
                        if test_features[n] > preffered_neurons[n]:
                            covered_neurons.append(n)
            coveredsize = len(covered_neurons)
            
            coverage = float(coveredsize) / len(topN)
#             covered_combinations = 0
#             max_comb = 0

        elif method == 'idc':
            topN = topN[: int(len(topN) * self.percentage)]
            
            nbr=min(self.user_input, len(topN))
            
            coverage, covered_combinations, max_comb = self.idc_pipline(self.model_name, train_inputs,
                                                                        self.subject_layer, test_inputs,
                                                                        self.selected_class, topN, subsetTop)
            covered_neurons = topN

        return coverage, len(covered_neurons), covered_combinations, max_comb

    def coverage_neurons(self, model1, model2):

        # Finding shared neurons between the two out and in domains
        model1_neurons = list(model1['max_activation_index'].unique())
        model2_neurons = list(model2['max_activation_index'].unique())
        
        initial_shared_neurons = find_shared_neurons(model1_neurons, model2_neurons)
        number_shared_neurons = len(initial_shared_neurons)
        

        neurons_inputs = {}
        scaledHD = {}
        hellinger_dict = {}
        scaled_hellinger = {}
        model1_pos_dict, model2_pos_dict = ({} for i in range(2))
        model1_features_list, model2_features_list = ([] for i in range(2))

        for neuron in initial_shared_neurons:
           
            model1_data = model1[model1['max_activation_index'] == neuron]
            model1_data["normalized_max_activations"] = model1_data["max_activations"].apply(
                lambda x: x / model1_data["max_activations"].sum())
            model2_data = model2[model2['max_activation_index'] == neuron]
            model2_data["normalized_max_activations"] = model2_data["max_activations"].apply(
                lambda x: x / model2_data["max_activations"].sum())

            # Getting all the unique features from both the models so the average can be taken.
            model1_dict, model2_dict, model3_dict = ({} for i in range(3))
            unique_features_model1 = model1_data['inputs'].unique()
            unique_features_model2 = model2_data['inputs'].unique()
            model1_pos_dict[neuron] = model1_data['inputs'].nunique()
            model2_pos_dict[neuron] = model2_data['inputs'].nunique()

            for feature in unique_features_model1:
                temp = model1_data[model1_data['inputs'] == feature]
                model1_dict[feature] = temp['normalized_max_activations'].mean()


            for feature in unique_features_model2:
                temp = model2_data[model2_data['inputs'] == feature]
                model2_dict[feature] = temp['normalized_max_activations'].mean()
                model2_features_list.append(model2_pos_dict[neuron])

            distance, num_features = hellinger_distance(model1_dict, model2_dict)
            hellinger_dict[neuron] = (distance, num_features)
            scaled_hellinger[neuron] = distance
            neurons_inputs[neuron] = num_features

        scaled_hellingerall = {key: scaled_hellinger[key] for key in initial_shared_neurons}


        save_KnwTrNeurons(scaled_hellinger, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'HDALL')
        
        avoided, gained, preferred = ([] for i in range(3))

        for neuron in initial_shared_neurons:
            if (self.distance_thershold-0.05)<scaled_hellinger[neuron] <=self.distance_thershold:
            # if (0.15) < scaled_hellinger[neuron] <= 0.24:
                if model1_pos_dict[neuron] > model2_pos_dict[neuron]:
                    avoided.append(neuron)
                elif model1_pos_dict[neuron] == model2_pos_dict[neuron]:
                    preferred.append(neuron)
                else:
                    gained.append(neuron)
        #
        save_KnwTrNeurons(gained, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'preferred')
        
        neurons_subset = [k for k in preferred if neurons_inputs[k] > 10]
        # print("neurons_subset", len(neurons_subset))
        preferred_subset = {key: hellinger_dict[key] for key in neurons_subset}
        preferred_neurons = {key: scaled_hellinger[key] for key in preferred}
        gained_neurons = {key: hellinger_dict[key] for key in gained}
        avoided_neurons = {key: hellinger_dict[key] for key in avoided}

        save_KnwTrNeurons(avoided_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'scaledHD')
       
        save_KnwTrNeurons(gained_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'Preffered_HD')
        
        save_KnwTrNeurons(gained_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'Gained_hellinger_distance')
        k_value = min(self.user_input, len(preferred_subset))
        top_10_neurons = heapq.nlargest(k_value, preferred_subset, key=preferred_subset.get)
       

        sub_set_N = []
        least_10_neurons = heapq.nsmallest(k_value, preferred_subset, key=preferred_subset.get)
        
        # get neuron_layer information
        knw_preffered_set, knw_gained_set, knw_top20_set, knw_least20_set = ([] for i in range(4))
        for g in gained:
            neuron_data = model1[model1['max_activation_index'] == g]
            input_index = neuron_data['inputs'].unique()
            knw_gained_set.extend(input_index)
        for p in preferred:
            l=neurons_inputs[p]
            neuron_data = model1[model1['max_activation_index'] == p]
            input_index = neuron_data['inputs'].unique()
            knw_preffered_set.extend(input_index)
       

        for n in top_10_neurons:
            neuron_data = model1[model1['max_activation_index'] == n]
            sub_set_N.append(neuron_data['neuron_index'])
            input_index = neuron_data['inputs'].unique()
            knw_top20_set.extend(input_index)
        for n in least_10_neurons:
            
            neuron_data = model1[model1['max_activation_index'] == n]
            
            input_index = neuron_data['inputs'].unique()
            knw_least20_set.extend(input_index)
        save_KnwTrNeurons(knw_top20_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'top20_testset')
        save_KnwTrNeurons(knw_least20_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'least20_testset')
        save_KnwTrNeurons(knw_preffered_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'VAL_set_preffered')

        save_KnwTrNeurons(knw_gained_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'topgained_testset')

       
        save_KnwTrNeurons(top_10_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'top20')

        save_KnwTrNeurons(least_10_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'least20')


        # SELECT NEURONS for COVERAGE Analysis per percentage of features length l: l higher better


        return initial_shared_neurons, avoided,top_10_neurons, least_10_neurons

    def idc_pipline(self, model_name, train_inputs, subject_layer, test_inputs, selected_class, Tr_knw_neuron,
                    subsetTop):
        covered_combinations = ()


        test_layer_outs = get_layer_outs_new(self.model, np.array(test_inputs), self.skip_layers)
       
        train_layer_outs = get_layer_outs_new(self.model, np.array(train_inputs), self.skip_layers)

        coverage, covered_combinations, max_comb = measure_idc(self.model, model_name,
                                                               test_inputs,
                                                               Tr_knw_neuron, subsetTop,
                                                               selected_class,
                                                               test_layer_outs, train_layer_outs, self.trainable_layers,
                                                               self.skip_layers,
                                                               covered_combinations)

        return coverage, covered_combinations, max_comb

    def load_data(self, statep):
       
        if self.data_set=='cifar_vgg':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_cifar_vgg()
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_cifar_vgg()
            img_rows, img_cols = 32, 32

        if self.data_set == 'mnist':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_MNISTVAL("fashion", channel_first=False)
                # X_train, y_train, X_test, y_test, X_val, y_val
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_MNISTVAL("ini", channel_first=False)
            img_rows, img_cols = 28, 28
        elif self.data_set == 'imagnet':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_Imagnet("zero", channel_first=False)
                # X_train, y_train, X_test, y_test, X_val, y_val
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_Imagnet("ini", channel_first=False)
            img_rows, img_cols = 224, 224

        elif self.data_set == 'cifar':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("zeroshot")
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("ini")
            img_rows, img_cols = 32, 32

        elif self.data_set == 'svhn':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_MNIST_32()

            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_SVHN_("ini")

            img_rows, img_cols = 32, 32

        return X_train, y_train, X_test, y_test, X_val, y_val, img_rows, img_cols

    def run(self, split, typTrknw, use_adv):
        # LOAD data

        print("loading data, ....")
        
        if self.data_set=='coco':
            X_train, y_train, X_test, y_test, X_val, y_val,X_val_z, y_val_z=load_coco()
            X_inpaint, y_inpaint, X_erase, y_erase= load_coco_augmented()
            img_rows, img_cols = 32, 32
        if self.data_set == 'leaves':
            print('leave data loaded')
            X_train, y_train, X_test, y_test, X_val, y_val, X_val_z, y_val_z  = load_leaves("original")
            img_rows, img_cols = 256, 256
            X_inpaint, y_inpaint, X_erase, y_erase, X_noise, y_noise = load_leaves('augmented')

        else:
            X_train, y_train, X_test, y_test, X_val, y_val, img_rows, img_cols = self.load_data("validation")
            X_train_z, y_train_z, X_test_z, y_test_z, X_val_z, y_val_z, img_rows_z, img_cols_z = self.load_data("zero_shot")
        
        if split > 0:
            split_x, split_y = split_data(split, X_test, y_test)
            # print("size of splited testset", split_x.shape)
            X_test, y_test = split_x, split_y


        
        print("knowlege transfer/change estimation, ...")
        
        print("for validation::", X_val.shape[0])
        print("for zero shot learning::", X_val_z.shape[0])
        
        try:
            print("Loading preferred neurons")
            top_10_neurons = load_KnwTrNeurons("experiments/{}_top20_knw.sav".format(self.data_set))
            least_10_neurons = load_KnwTrNeurons("experiments/{}_least20_knw.sav".format(self.data_set))
            preferred = load_KnwTrNeurons(
                "experiments/{}_preferred_knw.sav".format(self.data_set))  # dict::neurons+numberof feature
            preferred_neurons = load_KnwTrNeurons("experiments/{}_Preffered_HD_knw.sav".format(self.data_set))
        except Exception as e:
            print("Preferred neurons must be calculated. Doing it now!")
        nc_val = self.KnwTransfer(X_val, "validation")
            
        nc_zero = self.KnwTransfer(X_val_z, "zero_shot")
        
        zero_shot_data = pd.read_pickle(
                "data/zero_shot_{}_layers.sav".format(self.data_set))  # , sep='\t',error_bad_lines=False)
        valid_data = pd.read_pickle(
                "data/validation_{}_layers.sav".format(self.data_set))  # , sep='\t',error_bad_lines=False)

        print("transfer knowledge neurons extraction, ...")

        shared_neurons, preferred, top_20_neurons, least_20_neurons = self.coverage_neurons(valid_data, zero_shot_data)

        save_KnwTrNeurons(top_20_neurons, '%s/%s_'
                              % (experiment_folder, self.data_set) + 'top20')

        save_KnwTrNeurons(least_20_neurons, '%s/%s_'
                              % (experiment_folder, self.data_set) + 'least20')

        print("neurons data saved ")

        top_10_neurons = load_KnwTrNeurons("experiments/{}_top20_knw.sav".format(self.data_set))
        least_10_neurons = load_KnwTrNeurons("experiments/{}_least20_knw.sav".format(self.data_set))
        preferred = load_KnwTrNeurons("experiments/{}_preferred_knw.sav".format(self.data_set))#dict::neurons+numberof feature
        preferred_neurons=load_KnwTrNeurons("experiments/{}_Preffered_HD_knw.sav".format(self.data_set))
            # Test Coverage Estimation

        if use_adv:
           
            X_test_a = get_adv(self.attack, self.data_set, X_test, y_test)

            X_test = X_test_a
            # print("adversarial attacks size:", X_test_a.shape[0])
        if self.use_augment:
            if self.augment == "inpainting":
                X_test, y_test = X_inpaint, y_inpaint
            elif self.augment == "erase":
                X_test, y_test = X_erase, y_erase
            elif self.augment== 'noise':
                X_test, y_test = X_noise, y_noise


        nc_test = self.KnwTransfer(X_test, "testing")
        test_data = pd.read_pickle(
            "data/testing_{}_layers.sav".format(self.data_set))  # , sep='\t',error_bad_lines=False
       
        if typTrknw == 'top':
            TopN = top_10_neurons
        elif typTrknw == 'least':
            TopN = least_10_neurons
        elif typTrknw == 'preferred':
            TopN = preferred_neurons
        
        TopN = dict(sorted(TopN.items(), key=operator.itemgetter(1), reverse=False))
        TopN = list(TopN.keys())
        
        subsetTop = []
        Knw_coverage, number_covered, covered_combinations, max_comb = self.coverage_score(
            TopN, test_data, X_val, X_test, subsetTop)


        return Knw_coverage, number_covered, covered_combinations, max_comb, X_test.shape[0], X_val_z.shape[0], TopN
