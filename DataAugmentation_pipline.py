from PIL import Image
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from matplotlib import pyplot as plt
# from stablediffusion.IMAGIC import *
# from wand.image import Image
from utils import load_knw_data
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100
from Dataprocessing import load_MNISTVAL, load_CIFARVAL, load_driving_data, load_SVHN_, load_EMNIST, load_MNIST, \
    load_CIFAR, load_MNIST_32, split_data, load_Imagnet, load_cifar_vgg, load_coco, load_coco_augmented, load_leaves_V

from sklearn.model_selection import train_test_split
from tensorflow.keras import utils
import jsonpickle
from utils import *
from augmentation import augment_coco_imgs, augment_leaves_imgs, augment_from_folder,augment_ALLcoco,augment_CIFAR_imgs,augment_KNW_imgs
from json import JSONEncoder
from leaves_data_processing import *
import random as ran
from matplotlib import cm
import skimage.io as io
import urllib.request
import urllib
import csv
from Augmentation_techniques import *
# import cv2
from PIL import Image
from torchvision import transforms
import os
from tqdm import tqdm
# from einops import rearrange
import numpy as np
from pathlib import Path
import argparse
from coco_preprocess import *
from Data_processing_pipline import Datapipline
from datetime import datetime
from Augmentation_techniques.Masking import *
from Augmentation_techniques.InpaintingDifussionModel import *
from Realism_measures.SSIM import *
from Realism_measures.FID import *
from Augmentation_techniques.Caption_Enrichement_NLP import *

__version__ = 0.2
def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """
    text = 'Data AUgmentation pipline'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)

    # new command-line arguments
    parser.add_argument("-V", "--version",  help="show program version",
                        action="version", version="DeepFault %f" % __version__)
    parser.add_argument("-M", "--methodology", help="semantic augmentation methodology", choices=['Inpainting','Imagic'])
    parser.add_argument("-DS", "--dataset", help="The dataset to be used (mnist\
                        SVHN or cifar10).", choices=["mnist","cifar10","SVHN"])
    parser.add_argument("-Me", "--measure", help="the approach to be employed \
                            to measure similarity", choices=['SSIM', 'FID'])

    parser.add_argument("-Cap", "--Augmented_caption", help="the image caption")
    parser.add_argument("-K", "--iteration", help="nbre of iteration for augmenting an image.", type=int)
    parser.add_argument("-SS", "--seed_size", help="size of initial set of seed images.", type=int)
    # parser.add_argument("-AugCap", "--Augmented_caption", help="the augmented image caption")

    parser.add_argument("-LOG", "--logfile", help="path to log file")

    args = parser.parse_args()

    return vars(args)
def NLP_captioning():
    return 0
def save_image(Aug_img, base_file, approach):
    directory='./data/Augmented'
    name = base_file
    new_file = '{}''_''{}'+'.png'.format(name, approach)
    completeName =os.path.join(directory,new_file )
    print (type(Aug_img))
    plt.imsave(name, Aug_img)
    print("saved")



if __name__ == "__main__":
    args = parse_arguments()
    approach = args['methodology'] if args['methodology'] else 'Inpainting'

    nlp = spacy.load("en_core_web_sm")
    taxonomy = ['smiling', 'waving', 'talking', 'sleeping', 'siting', 'laughting', 'jumping', 'wearing a mask']
    measure = args['measure'] if not args['measure'] == None else 'SSIM'
    # caption = args['intial_caption'] if not args['intial_caption'] == None else 'a person'
    Augmented_caption = args['Augmented_caption'] if not args['Augmented_caption'] == None else 'a person'
    Seed_size = args['seed_size'] if not args['seed_size'] == None else 2
    dataset = args['dataset'] if not args['dataset'] == None else 'coco_animal'#['knw']
    # datatype = args['datatype'] if not args['datatype'] == None else 'cifar'
    logfile_name = args['logfile'] if args['logfile'] else 'DataAugment.log'
    logfile = open(logfile_name, 'a')
    extension='.csv'

    iteration="_11_"
    # Format as DATE - REGION - REPORT TYPE
    start_time = datetime.now()
    results=[]
    # line = [str(id), SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise, clip_score.item()]
    # results.append(line)
    # images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category, "original": ini_path,
    #                "Inpainting": inpaint_path, "Erase": erase_path, "Noise": noise_path})

    entete_results=['img_id', 'SSIM_Inp', 'SSIM_Erase','SSIM_Noise', 'FID_inp', 'FID_Erase','FID_Noise', 'Clip score']
    # device = torch.device("cpu")
    entete_imgs=["id", "caption", "Aug_caption", "original_img", "Inpainting", "Erase", "Noise", "label", "aug_category"]
    approach = 'Inpainting'

    date = datetime.strftime(datetime.now(), '%Y-%m-%d')
    info = {f'{date} - {approach} - {dataset} - {extension}'}
    extension = '.csv'
    k="scores_matadata"
    a="augmentedset"
    file_name = f'{date}-{approach}-{dataset}-{k}-{iteration}-{extension}'
    file_images=f'{date}-{approach}-{dataset}-{a}-{iteration}-{extension}'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # with torch.cuda.amp.autocast(True):
    #####dataset and input
    start_time = datetime.now()
    logfile_name = './results/DataAugment.log'
    logfile = open(logfile_name, 'a')
    if dataset == 'cifar':
        print(dataset,"loading... ... ...")
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        directory_aug_data = "./data/cifar/Augmented/"
        (X_train, y_train), (X_test, y_test) = cifar10.load_data()
        results, images = augment_CIFAR_imgs(Seed_size, X_test, y_test, approach, directory_aug_data, categories)

        print("size of augmented data set:", len(results))
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_results)
            for l in results:
                tsv_writer.writerow(l)
        with open('./results/' + file_images, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_imgs)
            for l in images:
                tsv_writer.writerow(l)

        print("saved csv")
    elif dataset=='coco':
        directory_aug_data="./data/"
        cat='person'
            # results, images = augment_coco_imgs(Seed_size, approach, directory_aug_data) <== worked fine
        results, images = augment_ALLcoco(Seed_size, directory_aug_data, cat)

        Augmented_data = {"info": info, "images": images}
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_results)
            for l in results:
                tsv_writer.writerow(l)
        with open('./results/' + file_images, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_imgs)
            for l in images:
                tsv_writer.writerow(l)
        print(len(results))
        print("saved csv")

        sampleJson = jsonpickle.encode(Augmented_data)
        with open('./results/augmented_instances_coco2017'+str(iteration)+'.json', 'w') as fp:
            json.dump(sampleJson, fp)
    elif dataset=='coco_animal':
        cat='animal'
        categories=['zebra',"bear","bird",'cat','cow']
        # ,]'dog','elephant','giraffe','horse','sheep'
        directory_aug_data = "./data/coco/Augmented/"
        testfolder = './dataset/coco/data_structured/train'
        train_metadata, val_metadata = load_annotation()
        categories_labels = get_category(train_metadata, cat)
        ImgsIds, categories_keys=get_info(categories_labels,train_metadata,val_metadata)
        # ImgsIds is a list
        # TestIds=get_ids_testset(categories, testfolder)

        results, images,inpainting_testing_data,erasing_testing_data,noise_testing_data = augment_from_folder(Seed_size, approach, directory_aug_data, testfolder,categories,ImgsIds, categories_keys)

        print("size of augmented data set:", len(results))
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_results)
            for l in results:
                tsv_writer.writerow(l)
        with open('./results/' + file_images, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_imgs)
            for l in images:
                tsv_writer.writerow(l)

        print("saved csv")
        save_coco_augmented_data(inpainting_testing_data, "inpainting")
        save_coco_augmented_data(erasing_testing_data, "erase")
        # save_coco_augmented_data(noise_testing_data, "noise")
        print("saved pickle")
    elif dataset == 'leaves':

        directory_aug_data="./data/Grapes/"
        testfile = './dataset/Grapes/test/x_test_coloured.pickle'
        ytest = './dataset/Grapes/test/y_test_coloured.pickle'

        X_test = load_data(testfile)
        Y_test = load_data(ytest)

        print("data loaded")
        taxonomy = 'grape '
        categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]
        training_data, testing_data=make_leaves_data()

        results, images = augment_leaves_imgs(Seed_size, testing_data, approach, directory_aug_data, taxonomy, categories)

        augmented_leaves= {"info": info, "images": images}
        print("size of augmented data set:",len(results))
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_results)
            for l in results:
                tsv_writer.writerow(l)
        # with open('./results/' + file_images, 'w') as out_file:
        #     tsv_writer = csv.writer(out_file, delimiter='\t')
        #     tsv_writer.writerow(entete_imgs)
        #     for l in images:
        #         tsv_writer.writerow(l)

        print("saved csv")
        sampleJson = jsonpickle.encode(augmented_leaves)
        with open('./results/augmented_instances_leaves'+str(iteration)+'.json', 'w') as fp:
            json.dump(sampleJson, fp)
        # with open(filename ,'w') as out_file:
        #     tsv_writer = csv.writer(out_file, delimiter='\t')
        #     tsv_writer.writerow(fields_to_gene)
        #     for l in lines:
        #         tsv_writer.writerow(l)

    elif dataset=='knw':

        directory_aug_data = "./data"
        datatype = 'Grapes'
        idd = 22000
        if datatype=='coco':
            X_train, y_train, X_test, y_test, X_val, y_val,X_val_z, y_val_z=load_coco()
            img_rows, img_cols = 32, 32
        elif datatype == 'Grapes':
            categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]
            X_train, y_train, X_test, y_test, X_val, y_val, X_test_z,y_test_z  = load_leaves_V("original")

        elif datatype == 'cifar':
            print(dataset, "loading... ... ...")
            categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
            directory_aug_data = "./data/cifar/Augmented/"
            X_train, y_train, X_test, y_test, X_val, y_val = load_CIFARVAL("ini")

        X_knw, y_knw=load_knw_data(X_val, y_val, "leaves")
        print("size of test knw preffered", X_knw.shape)



        scores, images=augment_KNW_imgs(Seed_size, X_knw, y_knw, approach, directory_aug_data, categories, datatype, idd)
        entete_r=['id', 'SSIM_inpaint', 'FID_inpainting', 'clip_score']
        with open('./results/' + file_name, 'w') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerow(entete_r)
            for l in results:
                tsv_writer.writerow(l)
    print("--- %s seconds ---" % (datetime.now() - start_time))





