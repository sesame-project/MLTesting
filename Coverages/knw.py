import csv
import numpy as np
from utils import *
import statistics
import pandas as pd
pd.options.mode.chained_assignment = None
import os
import pickle
from collections import defaultdict
from Coverages.idc_knw import *
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.datasets import mnist, fashion_mnist, cifar10
from keras.preprocessing import image
from Coverages.idc import find_relevant_neurons
from utils_ini import save_quantization, load_quantization, save_totalR, load_totalR
from keras.models import model_from_json
from keras.models import model_from_yaml
from datetime import datetime
from keras.models import model_from_json, load_model, save_model
from keras.layers import Input
from keras.utils import np_utils
from keras import models
from itertools import chain
from Dataprocessing import load_MNISTVAL, load_CIFARVAL, load_driving_data, load_SVHN_, load_EMNIST, load_MNIST, load_CIFAR, load_MNIST_32, split_data
from hellinger_distance import *

from lrp_toolbox.model_io import write, read
def scaleHD(points,val, rmax=1, rmin=0):
    maxval=points[max(points, key=points.get)]

    minval=points[min(points, key=points.get)]

    X_std = (float(val) - minval) / (maxval - minval)
    # X_scaled = X_std * (rmax - rmin) + rmin
    # print("the scaled value", val)
    return X_std
def get_adv(attack, dataset, X_test, Y_test):

        # adv_image_all = []
        # adv_label_all = []
        #
        # adv_image_all.append(np.load('dataset/adv_image/{}_{}_image.npy'.format(attack, dataset)))
        # adv_label_all.append(np.load('dataset/adv_image/{}_{}_label.npy'.format(attack, dataset)))
        #
        # adv_image_all = np.concatenate(adv_image_all, axis=0)
        # adv_label_all = np.concatenate(adv_label_all, axis=0)
        # # print(Y_test.shape)
        # # print(adv_label_all.shape)
        # test = np.concatenate([X_test, adv_image_all], axis=0)
        # true_test = np.concatenate([Y_test, adv_label_all], axis=0)
        adv_image_all = []
        adv_label_all = []
        attack_lst=[attack]
        # for att in attack_lst:
        adv_image_all.append(np.load('dataset/adv_image/{}_{}_image.npy'.format(attack, dataset)))
            # adv_label_all.append(np.load('dataset/adv_image/{}_{}_label.npy'.format(att, dataset)))

        # # print(adv_label_all.shape)
        adv_image_all = np.concatenate(adv_image_all, axis=0)
        # adv_label_all = np.concatenate(adv_label_all, axis=0)
        # print("Xtest shape",X_test.shape)
        # print("adv image",adv_image_all.shape)
        test = np.concatenate([X_test[:int(len(X_test)*0.5)], adv_image_all], axis=0)
        # true_test = np.concatenate([Y_test, adv_label_all], axis=0)
        return test
def save_KnwTrNeurons(neurons, filename):
    filename = filename + '_knw.pkl'
    # with h5py.File(filename, 'a') as hf:
    #     group = hf.create_group('group' + str(group_index))
    #     for i in range(len(neurons)):
    #         group.create_dataset("trneurons_" + str(i), data=neurons[i])
    with open(filename, 'wb') as out_file:
        pickle.dump(neurons, out_file)

    # print("total knw neurons data saved in ", filename)
    return


def load_KnwTrNeurons(filename):

        neurons= pd.read_pickle(filename)  # , sep='\t',error_bad_lines=False)

        return neurons


class KnowledgeCoverage:

    def __init__(self, model, data_set, model_name, subject_layer, trainable_layers, dense_layers,method, percent, thershold, attack, scaler=default_scale,  skip_layers=None,selected_class=1 ):
        self.activation_table = defaultdict(bool)
        self.dense_layers=dense_layers
        self.model = model
        self.attack = attack
        self.scaler = scaler
        self.data_set = data_set
        self.skip_layers = skip_layers = ([] if skip_layers is None else skip_layers)
        self.model_name= model_name
        self.subject_layer= subject_layer
        self.trainable_layers=trainable_layers
        self.selected_class=selected_class
        self.method=method
        self.percentage=percent
        self.distance_thershold=thershold

    def KnwTransfer(self, test_inputs, statep):
        """
               :param test_inputs: Inputs
               :return: Tuple containing the coverage and the measurements used to compute the coverage. 0th element is the
               percentage neuron coverage value.
               """
        saved_dir= "data/"+statep+'_'+self.data_set+'_layers.pkl'
        fields_to_generate=['inputs', 'activations', 'max_activations', 'max_activation_index','neuron_index']
        data=[]

        # with open( saved_dir, 'w') as out_file:
        #     tsv_writer = csv.writer(out_file, delimiter='\t')
        #     tsv_writer.writerow(fields_to_generate)

        # outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)

        outs = get_layer_outs_new(self.model, test_inputs, self.skip_layers)
        inp_cnt = 0
        activation_trace = {}
        # network_info=[]
        for layer_index, layer_out in enumerate(outs):  # layer_out is output of layer for all inputs
            input_inx = 0

            # print("layer!!!!!!!!!!!!",layer_index)
            if layer_index in self.trainable_layers:
                for out_for_input in layer_out:  # out_for_input is output of layer for single input
                    #out_for_input = self.scaler(out_for_input)
                    # print("input", input_inx)
                    # print(out_for_input.shape)
                    total_neuron = out_for_input.shape[-1]
                    # network_info.append((total_neuron, layer_index))
                    for neuron_index in range(out_for_input.shape[-1]):
                        maxact = np.max(out_for_input[..., neuron_index])
                        activation = (out_for_input[..., neuron_index])
                        # neuron_arg=np.argmax(out_for_input.squeeze(0))
                        global_neuron_index = (layer_index, neuron_index)
                        activation_trace[input_inx]=(activation,neuron_index)
                        # ['inputs', 'activations', 'max_activations', 'max_activation_index']
                        data.append([input_inx,activation, maxact,global_neuron_index, neuron_index])


                    input_inx += 1
                """ on the 09-03-2022 changed
                            for layer_index, layer_activation in enumerate(outs):  # layer_activation is output of layer for all inputs
                                # print("THE LAYER INDEX !!!!!",layer_index)
                
                                if layer_index in self.trainable_layers:
                                    n_features = layer_activation.shape[
                                        -1]  # Number of features in the feature map #The feature map has shape (1, size, size, n_features).
                                    # print("layer:", layer_index)
                
                                    activation = list(layer_activation.squeeze(0))
                                    input_feat = feature
                                    out_for_input = layer_activation[0]
                                    neuron = np.argmax(activation)
                                    max_act = np.max(activation)
                                    inp_cnt["features"] = input_feat
               """

        # print(network_info)
        # flipped = {}
        # all=[]
        # for key, value in activation_trace.items():
        #     if key not in flipped:
        #         flipped[key] = [value]
        #     else:
        #         flipped[key].append(value)
        # for key, value in flipped.items():
        #     flipped[key]=np.array(value)
        # for key, value in flipped.items():
        #     maxact = np.max(value)
        #     neuron_arg = np.argmax(value)
        #     # printing result
        #     print("value: " ,value)
        #     all.append([key,value,maxact,neuron_arg])
        #     data.append([key,value,maxact,neuron_arg])
        # print(len(all))


        df=pd.DataFrame(data,
                   columns=fields_to_generate)
        # print("frame created")
        with open(saved_dir, 'wb') as out_file:
            pickle.dump(df, out_file)


        return data

    def coverage_score(self,  topN, testdata, train_inputs, test_inputs, subsetTop):

        covered_neurons = []
        testmodel=testdata
        # print(type(testmodel))
        method=self.method
        # testmodel['max_activation_index'] = pd.to_numeric(testmodel['max_activation_index'][1])
        # testmodel["max_activations"] = pd.to_numeric(testmodel['max_activations'])
        # testmodel['inputs'] = pd.to_numeric(testmodel['inputs'])
        modeltest_neurons = list(testmodel['max_activation_index'].unique())
        Trknw_shared_neurons = find_shared_neurons(modeltest_neurons, topN)
        number_Trknw_shared_neurons = len(Trknw_shared_neurons)
        # print("number of neuron after the test", number_Trknw_shared_neurons)
        test_features={}
        for neuron in Trknw_shared_neurons:
                model_data = testmodel[testmodel['max_activation_index'] == neuron]
                model_data["normalized_max_activations"] = model_data["max_activations"].apply(
                    lambda x: x / model_data["max_activations"].sum())

                # Getting all the unique features.
                model_dict= {}
                unique_features = model_data['inputs'].unique()
                test_features[neuron]=len(list(unique_features))

        if method == 'feature_l':
                # for n in topN:
                #         if n in test_features:
                #             if test_features[n] > preffered_neurons[n]:
                #                 covered_neurons.append(n)
                # coveredsize = len(covered_neurons)
                #
                # coverage = float(coveredsize) / len(topN)
                covered_combinations=0
                max_comb=0

        elif method == 'idc':
            topN= topN[: int(len(topN) * self.percentage)]
            coverage, covered_combinations, max_comb = self.idc_pipline(self.model_name, train_inputs,
                                                                   self.subject_layer, test_inputs,
                                                                   self.selected_class, topN, subsetTop)
            covered_neurons=topN

        return coverage , len(covered_neurons), covered_combinations,max_comb

    def coverage_neurons(self, model1, model2):

        # Finding shared neurons between the two out and in stages/models
        model1_neurons = list(model1['max_activation_index'].unique())
        model2_neurons = list(model2['max_activation_index'].unique())
        # model1_neurons_global = list(model1['global_index'].unique())
        # model2_neurons_global = list(model2['global_index'].unique())
        print("model 1 neurons size",len(model1_neurons))
        initial_shared_neurons = find_shared_neurons(model1_neurons, model2_neurons)
        number_shared_neurons = len(initial_shared_neurons)
        # global_shared=find_shared_neurons(model1_neurons_global, model2_neurons_global)
        # print("global neurons size",len(global_shared))
        shared_dist = {}
        initial_shared = {}
        scaledHD={}
        hellinger_dict={}
        hellingerNN_average={}
        Average_hellinger_dict={}
        model1_pos_dict, model2_pos_dict = ({} for i in range(2))
        # print("initial shared ones", len(initial_shared_neurons))
        model1_features_list, model2_features_list = ([] for i in range(2))

        for neuron in initial_shared_neurons:
            # Loading the data for both models
            model1_data = model1[model1['max_activation_index'] == neuron]
            model1_data["normalized_max_activations"] = model1_data["max_activations"].apply(
                lambda x: x / model1_data["max_activations"].sum())
            model1_data["neuron_index"]=model1_data["neuron_index"]
            model2_data = model2[model2['max_activation_index'] == neuron]
            model2_data["normalized_max_activations"] = model2_data["max_activations"].apply(
                lambda x: x / model2_data["max_activations"].sum())
            model2_data["neuron_index"] = model2_data["neuron_index"]

            # Getting all the unique features from both the models so the average can be taken.
            model1_dict, model2_dict, model3_dict = ({} for i in range(3))
            unique_features_model1 = model1_data['inputs'].unique()
            unique_features_model2 = model2_data['inputs'].unique()
            model1_pos_dict[neuron] = model1_data['inputs'].nunique()
            model2_pos_dict[neuron] = model2_data['inputs'].nunique()

            for feature in unique_features_model1:
                temp = model1_data[model1_data['inputs'] == feature]
                model1_dict[feature] = temp['normalized_max_activations'].mean()
                model1_features_list.append(model1_pos_dict[neuron])

            for feature in unique_features_model2:
                temp = model2_data[model2_data['inputs'] == feature]
                model2_dict[feature] = temp['normalized_max_activations'].mean()
                model2_features_list.append(model2_pos_dict[neuron])

            """Quantify knowledge change:
            quantify neuron feature preference shifts
            to measure per - neuron knowledge change during zeroshot transfer
            """
            """
            # features length l: num_features describes the number of(unique) maximally
            # activated features in a feature preference distribution (model1_dict)

            # analyze zeroshot transfer as distribution shifts
            #high distance between the same neuron in different models tell us that the pretrained neuron
            #did not abstract the new domain image/feature well
            """

            # Hellinger Dictionary contains the distance between two model for a given activation and total number of
            # features compared between these two models.

            distance, num_features = hellinger_distance(model1_dict, model2_dict)

            hellinger_dict[neuron] = (distance, num_features)
            Average_hellinger_dict[neuron] = (distance, num_features)

            hellingerNN_average[neuron] = (distance * num_features)
            initial_shared[neuron] = num_features




        """New feature selection"""
        #"""
        # for neuron in list(model1['max_activation_index'].unique()):
        #     model1_data = model1[model1['max_activation_index'] == neuron]
        #     model1_pos_dict[neuron] = model1_data['inputs'].nunique()
        # for neuron in list(model2['max_activation_index'].unique()):
        #     model2_data = model2[model2['max_activation_index'] == neuron]
        #     model2_pos_dict[neuron] = model2_data['inputs'].nunique()

        save_KnwTrNeurons(hellinger_dict, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'hellinger_stats')
        avoided, gained, preferred= ([] for i in range(3))

        for neuron in initial_shared_neurons:
            # print(model2_data[model1_data['max_activation_index'] == neuron])
            if model1_pos_dict[neuron] > model2_pos_dict[neuron]:
                avoided.append(neuron)
            elif model1_pos_dict[neuron] == model2_pos_dict[neuron]:
                preferred.append(neuron)
            else:
                gained.append(neuron)
        #
        # print("avoided",len(avoided))
        # print("\n")
        # print("shared", len(preferred))
        # print("\n")
        # print("gained", len(gained))
        # print("\n")
        """For shared"""
        # neurons_subset={}
        # H_distance={key: hellinger_dict[key] for key in preferred}
        # top_10_neurons = heapq.nlargest(10, H_distance, key=H_distance.get)
        # top_20_neurons = heapq.nlargest(20, H_distance, key=H_distance.get)
        # least_20_neurons = heapq.nsmallest(20, H_distance, key=H_distance.get)
        # least_10_neurons = heapq.nsmallest(10, H_distance, key=H_distance.get)
        # #get neuron_layer information
        # for p in preferred:
        #         print("the neuron", p)
        #         #model1.set_index('max_activation_index', inplace=True)
        #         neuron_data = model1[model1['max_activation_index'] == p]
        #         layer = neuron_data['layer_index'].unique()
        #         print(layer)
        #         neurons_subset.update({p: layer})
        neurons_subset = []
        # first filtering based on numfeature
        neurons_subset = [k for k in initial_shared if initial_shared[k] >0]
        H_distance = {key: hellingerNN_average[key] for key in neurons_subset}
        # print("H_distance_not average before nan",len(H_distance))



        for key, value in H_distance.items():
            if not pd.isna(float(value)):
                # print("value!!",value)
                scaledHD[key]=value
        maxval = scaledHD[max(scaledHD, key=scaledHD.get)]
        minval = scaledHD[min(scaledHD, key=scaledHD.get)]
        # print("the min and max", minval,maxval)
        for key, value in scaledHD.items():
                scaledvalue = scaleHD(scaledHD, value)
                scaledHD[key] = scaledvalue
        # print("H_distance after scale ", len(scaledHD))

        save_KnwTrNeurons(scaledHD, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'HellingerDistancescaled')#non averaged HD values


        top_10_neurons = heapq.nlargest(10, scaledHD, key=scaledHD.get)
        # print(top_10_neurons)

        top_20_neurons = heapq.nlargest(20, scaledHD, key=scaledHD.get)
        # for n in top_20_neurons:
            # print(n)
            # print(scaledHD[n])

        least_20_neurons = heapq.nsmallest(20, H_distance, key=H_distance.get)
        least_10_neurons = heapq.nsmallest(10, scaledHD, key=scaledHD.get)
        print(least_10_neurons)
        # for n in least_10_neurons:
            # print(n)
            # print(scaledHD[n])

        # get neuron_layer information
        knw_preffered_set,knw_gained_set, knw_top20_set,knw_least20_set = ([] for i in range(4))
        for g in gained:

            neuron_data = model1[model1['max_activation_index'] == g]
            input_index=neuron_data['inputs'].unique()
            knw_gained_set.extend(input_index)
        for p in preferred:
                neuron_data = model1[model1['max_activation_index'] == p]
                input_index = neuron_data['inputs'].unique()
                knw_preffered_set.extend(input_index)
            #print(layer)

        for n in top_10_neurons:
                # model1.set_index('max_activation_index', inplace=True)
                neuron_data = model1[model1['max_activation_index'] == n]
                # layer = n[0]
                # print(layer)
                input_index=neuron_data['inputs'].unique()
                knw_top20_set.extend(input_index)
        for n in least_10_neurons:
                # model1.set_index('max_activation_index', inplace=True)
                neuron_data = model1[model1['max_activation_index'] == n]
                # layer = n[0]
                # print(layer)
                input_index=neuron_data['inputs'].unique()
                knw_least20_set.extend(input_index)
        save_KnwTrNeurons(knw_top20_set, '%s/%s_'
                                  % (experiment_folder, self.data_set) + 'top20_testset')
        save_KnwTrNeurons(knw_least20_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'least20_testset')
        save_KnwTrNeurons(knw_preffered_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'toppreferred_testset')
        save_KnwTrNeurons(knw_gained_set, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'topgained_testset')
        # print(neurons_subset)
        # a_subset = {key: dict_prefer[key] for key in topN}
        save_KnwTrNeurons(top_10_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'top20')

        save_KnwTrNeurons(least_10_neurons, '%s/%s_'
                          % (experiment_folder, self.data_set) + 'least20')



        """Old features selection
        #################################################
        # print("topN", len(top_10_neurons))
        # stats
        median_H_D = np.median(list(shared_dist.values()))
        median_l=statistics.median(list(shared.values()))
        ##############Apply thersholds for H.distance and activation>0 and number features
        
        condition that it does belong to the trainable layers'''
        #distance filtering
        preffered_neurons = {k: v for (k, v) in shared_dist.items() if v <= self.distance_thershold}
        preffered_fil={}
        dict_prefer={}
        for n in initial_shared_neurons:
            l=0
            if n in model1_neurons:
                neuron_info = model1_data[model1_data['max_activation_index'] == neuron]
                layer = neuron_info['layer_index'].unique()
                max_actv= neuron_info['max_activations']

                for act in max_actv:
                    if act > 0:
                        l+= 1
            # case={'features_length': l,'layer':layer}
                        preffered_fil[n] =l
                        dict_prefer.update({n: layer})
        """
        # SELECT NEURONS for COVERAGE Analysis per percentage of features length l: l higher better
        """ Changed on 09-03-2022
        #############################################################
        #############################################################
        preffered = {}
        subset = []
        preffered_diction = [k for k in preffered_fil if preffered_fil[k] >= median_l]

        for k, v in preffered_fil.items():
            if k in preffered_diction:
                preffered[k] = v
       
        sortedpreffered = sorted(preffered, key=preffered.get, reverse=True)
        topN = sortedpreffered[: int(len(sortedpreffered) * self.percentage)]

        # a_subset = {key: dict_prefer[key] for key in sortedpreffered}
        a_subset = {key: dict_prefer[key] for key in topN}
        
        print("topN final",len(topN))
        #############################################################
        #############################################################
        """
        #############################################################
        #############################################################

        """
        preffered = {}
        for k, v in preffered_fil.items():
            if k in preffered_neurons:
                preffered[k] = v
        print("preffered length based on both median and length", len(preffered))

        sortedpreffered = sorted(preffered, key=preffered.get, reverse=True)
        topN = sortedpreffered[: int(len(sortedpreffered) * self.percentage)]

        # a_subset = {key: dict_prefer[key] for key in sortedpreffered}
        a_subset = {key: dict_prefer[key] for key in topN}
        print("topN final", len(topN))
        """

        return initial_shared_neurons, preferred,top_10_neurons, least_10_neurons


    def idc_pipline(self, model_name, train_inputs, subject_layer, test_inputs, selected_class, Tr_knw_neuron, subsetTop):
        covered_combinations = ()
        num_Tr_neurons=len(Tr_knw_neuron)
        try:
            print("Loading unsupervised clustering results")
            qtized = load_quantization('%s/%s_%d_%d_%d_silhouette'
                                %(experiment_folder,
                                self.model_name,
                                self.selected_class,
                                self.subject_layer,
                                num_Tr_neurons),0)
        except:
            print("Clustering results NOT FOUND; Calculating them now!")
            if 'conv' in self.model.layers[subject_layer].name:
                    is_conv = True
            else:
                    is_conv = False

            train_layer_outs = get_layer_outs_new(self.model, np.array(train_inputs))

            qtized, clusters= quantizeSilhouette(train_layer_outs, is_conv,
                                            Tr_knw_neuron,self.model, subsetTop)
            # print("the qtized")
            # print(qtized)



        test_layer_outs = get_layer_outs_new(self.model, np.array(test_inputs))

        coverage, covered_combinations, max_comb = measure_idc(self.model, model_name,
                                                               test_inputs, subject_layer,
                                                               Tr_knw_neuron,subsetTop,
                                                               selected_class,
                                                               test_layer_outs, qtized, is_conv,
                                                               covered_combinations)

        return coverage, covered_combinations, max_comb

    def load_data(self, statep):

        if self.data_set == 'mnist':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_MNISTVAL("fashion", channel_first=False)
                # X_train, y_train, X_test, y_test, X_val, y_val
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_MNISTVAL("ini", channel_first=False)
            img_rows, img_cols = 28, 28

        elif self.data_set == 'cifar':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("zeroshot")
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("ini")
            img_rows, img_cols = 32, 32

        elif self.data_set == 'svhn':
            if statep == 'zero_shot':
                X_train, y_train, X_test, y_test = load_MNIST_32()
                X_val=[]
                y_val=[]
            else:
                X_train, y_train, X_test, y_test, X_val, y_val = load_SVHN_("ini")


            img_rows, img_cols = 32, 32





        return X_train, y_train, X_test, y_test, X_val,y_val, img_rows, img_cols



    def run(self, split,typTrknw,use_adv):
        #LOAD data
        print("loading data, ....")
        X_train, y_train, X_test, y_test, X_val, y_val ,img_rows, img_cols =self.load_data("validation")
        X_train_z, y_train_z, X_test_z, y_test_z, X_val_z, y_val_z, img_rows_z, img_cols_z = self.load_data("zero_shot")
        if split>0:
            split_x, split_y=split_data(split, X_test, y_test)
            print("size of splited testset", split_x.shape)

            X_test, y_test=split_x, split_y



        #Knowledge change estimation
        print("knowlege transfer/change estimation, ...")
        print("size of the ", self.data_set, "data used:")
        print("for validation", X_val.shape)
        print("for zero shot learning", X_test_z.shape)
        # print("initial test set size::",X_test.shape)

        # nc_val=self.KnwTransfer(X_val, "validation")
        print("validation done,...")
        #nc_zero=self.KnwTransfer(X_test_z, "zero_shot")
        print("zeroshot done,...")




        # #filename = os.path.join("data", "zshot_valid_in_layers_{}_.pickle".format(self.data_set))
        # zero_shot_data = pd.read_pickle("data/zero_shot_{}_layers.pkl".format(self.data_set))#, sep='\t',error_bad_lines=False)
        #
        # valid_data = pd.read_pickle("data/validation_{}_layers.pkl".format(self.data_set))#, sep='\t',error_bad_lines=False)
        #
        #
        # print("transfer knowledge neurons extraction, ...")
        #
        # shared_neurons,preferred,top_20_neurons, least_20_neurons= self.coverage_neurons(valid_data, zero_shot_data)
        # #
        # #
        # #
        # save_KnwTrNeurons(top_20_neurons, '%s/%s_'
        #                  % (experiment_folder, self.data_set) + 'top20')
        #
        # save_KnwTrNeurons(least_20_neurons, '%s/%s_'
        #                  % (experiment_folder, self.data_set) + 'least20')


        print("neurons data saved ")
        if use_adv:
            print(use_adv)
            print("Adversarials attacks::::", self.attack)
            # print(X_test.shape)
            X_test_a= get_adv(self.attack, self.data_set, X_test, y_test)
            X_test=X_test_a
            print("adversarial attacks shape:", X_test_a.shape)
        else:
            X_test = X_test


        top_10_neurons=load_KnwTrNeurons("experiments/{}_top20_knw.pkl".format(self.data_set))
        least_10_neurons = load_KnwTrNeurons("experiments/{}_least20_knw.pkl".format(self.data_set))
            #Test Coverage Estimation

        # print("test set to be used ", X_test.shape[0])
        nc_test = self.KnwTransfer(X_test, "testing")
        test_data = pd.read_pickle("data/testing_{}_layers.pkl".format(self.data_set))#, sep='\t',error_bad_lines=False
        # print("load tesdata ok")
        if typTrknw=='top':
            TopN=top_10_neurons
        elif typTrknw=='least':
            TopN=least_10_neurons
        subsetTop=[]

        print("test set coverage estimation, ...")
        Knw_coverage, number_covered,covered_combinations, max_comb = self.coverage_score(
                     TopN, test_data, X_val, X_test, subsetTop)



        return Knw_coverage, number_covered, covered_combinations, max_comb, X_test.shape[0],X_test_z.shape[0],TopN

        # preferred, number_shared_neurons, neurons_subset, top_10_neurons, top_20_neurons, least_10_neurons, least_20_neurons

        #Knw_coverage, number_covered, numberpreferred_neurons, covered_combinations, max_comb, X_test.shape[0],X_test_z.shape[0],numberpreffered_distance
