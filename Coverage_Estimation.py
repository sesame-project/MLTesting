import argparse
# from absl import app
# from absl import flags
import tensorflow as tf
from tensorflow import keras

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
from Coverages.knw import *
from tensorflow.keras import applications
from tensorflow.python.client import device_lib

__version__ = 0.9


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled

def by_indices(outs, indices):
    return [[outs[i][0][indices]] for i in range(len(outs))]


def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """

    # define the program description
    text = 'Knowledge Coverage Comparative studies for DNNs'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # add new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--model", help="Path to the model to be loaded.\
                        The specified model will be used.")#, required=True)
                        # choices=['lenet1','lenet4', 'lenet5'], required=True)
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN"])#, required=True)
    parser.add_argument("-A", "--approach", help="the approach to be employed \
                        to measure coverage", choices=['knw','idc','nc','kmnc',
                        'nbc','snac','tknc','ssc', 'lsa', 'dsa'])
    parser.add_argument("-C", "--class", help="the selected class", type=int)
    parser.add_argument("-Q", "--quantize", help="quantization granularity for \
                        combinatorial other_coverage_metrics.", type= int)
    parser.add_argument("-L", "--layer", help="the subject layer's index for \
                        combinatorial cov. NOTE THAT ONLY TRAINABLE LAYERS CAN \
                        BE SELECTED", type= int)
    parser.add_argument("-KS", "--k_sections", help="number of sections used in \
                        k multisection other_coverage_metrics", type=int)
    parser.add_argument("-KN", "--k_neurons", help="number of neurons used in \
                        top k neuron other_coverage_metrics", type=int)
    parser.add_argument("-RN", "--rel_neurons", help="number of neurons considered\
                        as relevant in combinatorial other_coverage_metrics", type=int)
    parser.add_argument("-AT", "--act_threshold", help="a threshold value used\
                        to consider if a neuron is activated or not.", type=float)
    parser.add_argument("-R", "--repeat", help="index of the repeating. (for\
                        the cases where you need to run the same experiments \
                        multiple times)", type=int)
    parser.add_argument("-LOG", "--logfile", help="path to log file")
    parser.add_argument("-ADV", "--advtype", help="path to log file")


    # parse command-line arguments


    # parse command-line arguments
    args = parser.parse_args()

    return vars(args)
def generate_coverage(dataset,percent, attack,modelpath, TypeTrknw,split):
    model_path = 'Networks/'+modelpath
    args = parse_arguments()
    approach = args['approach'] if args['approach'] else 'knw'
    num_rel_neurons = args['rel_neurons'] if args['rel_neurons'] else 2
    threshold = args['act_threshold'] if args['act_threshold'] else 0.5

    k_sect = args['k_sections'] if args['k_sections'] else 1000
    selected_class = args['class'] if not args['class'] == None else -1  # ALL CLASSES

    logfile_name = args['logfile'] if args['logfile'] else 'result_mnist.log'
    quantization_granularity = args['quantize'] if args['quantize'] else 3

    logfile = open(logfile_name, 'a')
    img_rows, img_cols, img_channel = 32, 32, 3
    ####################
    # 1) Setup the model
    model_name = model_path.split('/')[-1]
    if model_name == 'vgg16':
        model = applications.VGG16(weights='imagenet', include_top=False,
                                   input_shape=(img_rows, img_cols, img_channel))
        print("Model VGG 16 is loaded")
    else:
        #
        # model_name = model_path.split('/')[-1]

        try:
            json_file = open(model_path + '.json', 'r')  # Read Keras model parameters (stored in JSON file)
            file_content = json_file.read()
            json_file.close()

            model = model_from_json(file_content)
            model.load_weights(model_path + '.h5')

            # Compile the model before using
            model.compile(loss='categorical_crossentropy',
                          optimizer='adam',
                          metrics=['accuracy'])
        except:
            print("exeception")
            model = load_model(model_path + '.hdf5')

        # 2) Load necessary information
    trainable_layers = get_trainable_layers(model)
    dense_layers=get_dense_layers(model)
    non_trainable_layers = list(set(range(len(model.layers))) - set(trainable_layers))
    print('Trainable layers: ' + str(trainable_layers))
    print('Non trainable layers: ' + str(non_trainable_layers))

    experiment_folder = 'experiments'

    # Investigate the penultimate layer
    subject_layer = args['layer'] if not args['layer'] == None else -1
    subject_layer = trainable_layers[subject_layer]

    skip_layers = [0]  # SKIP LAYERS FOR NC, KMNC, NBC
    for idx, lyr in enumerate(model.layers):
        if 'flatten' in lyr.__class__.__name__.lower(): skip_layers.append(idx)

    print("Skipping layers:", skip_layers)
    ####################

    if approach == 'knw':
        model_folder = 'Networks'
        method = 'idc'
        # attack_lst = ['cw', 'fgsm', 'jsma', 'bim']
        knw = KnowledgeCoverage(model, dataset, model_name, subject_layer, trainable_layers,dense_layers, method, percent, threshold,attack, skip_layers=None, selected_class=1)



        # for attack in attack_lst:
        #knw_coverage, covered_TrKnw, TrKnw_neurons, combinations, max_comb, testsize, zero_size, preffered_distance \
        Knw_coverage, covered_TrKnw, combinations, max_comb, testsize, zero_size,Trkneurons= knw.run(split,TypeTrknw,use_adv=False)



        print("The model Transfer Knowledge Neurons number: ", covered_TrKnw)
        # print("type",TypeTrknw)
        print("The percentage of the used neurons out of all Transfer Knowledge Neurons : ",percent)

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

    #     # print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    #     # tf.debugging.set_log_device_placement(True)
    #     print(tf.config.list_physical_devices('GPU'))
    #     print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
    #     print(tf.test.is_built_with_cuda())
    #     print(device_lib.list_local_devices())
    sys_details = tf.sysconfig.get_build_info()
    cuda_version = sys_details["cuda_version"]
    print(cuda_version)
    #generate attacks for svhn
    with tf.device('/CPU:0'):
        startTime = time.time()

        split_lst=[0]#,0.25,0.5,0.75]
        data_lst = ['svhn']
        percent_lst = [0.5,0.6]
        TypeTrknw=['top'] #
        attack=['mim','bim','pgd','fgsm']
        filename = "svhn_attack.csv"#"KNW_split_new_idc.csv"
        fields_to_gene = ["model", "dataset", "test_size", "zeroshot", "coverage", "knw_neurons", "type knwTr neurons",
                         "split", "combinations covered",
                         "total combinations", "tkwn neurons", "attack"]
        lines=[]
        for dataset in data_lst:
            if dataset == "mnist":
                    mdl = 'LeNet1'
            elif dataset == 'svhn':
                    mdl = 'model_svhn'
            elif dataset == 'cifar':
                    mdl = 'model_cifar10'
            for tp in TypeTrknw :

                    for percent in percent_lst:
                        for split in split_lst:
                           for att in attack:

                            line=generate_coverage(dataset, percent,att,mdl,tp,split)
                            lines.append(line)


        # pool.close()
        # pool.join()
        with open(filename ,'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(fields_to_gene)
            for l in lines:
                tsv_writer.writerow(l)

        endTime = time.time()
        elapsedTime = endTime - startTime
        print("Elapsed Time = %s" % elapsedTime)
    # time.sleep(TIMEOUT)
    # Place tensors on the CPU






