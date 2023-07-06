from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
import tensorflow as tf
import numpy as np
import tarfile
import re as r
import os
import matplotlib.pyplot as plt
import time
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pickle
import random

from sklearn.model_selection import train_test_split

def select_valid_seed(filecv, thresholdCLIP, thresholdSSIM):
    dataset = pd.read_csv(open(filecv, 'rb'), delimiter='\t')

    print("dataframe len",len(dataset))

    stats = dataset[['SSIM_Inp', 'FID_inp', 'Clipscore']].describe()

    valid_CLIP_Inp = dataset[dataset.Clipscore >= 0.8]
    print("valid_CLIP_Inp",len(valid_CLIP_Inp))
    columns = ['img_id' , 'SSIM_Inp' , 'SSIM_Erase',  'SSIM_Noise', 'FID_inp' , 'FID_Erase', 'FID_Noise', 'Clipscore']
    dataset = dataset.reindex(columns=columns)

    data = dataset[['img_id' , 'SSIM_Inp' , 'SSIM_Erase',  'SSIM_Noise', 'Clipscore']]
    data = data.dropna(subset=['img_id'])
    inpaintingClip = []
    inpaintingssim = []
    erasedata = []
    noisedata = []
    for index, rows in data.iterrows():
        if rows.Clipscore >= thresholdCLIP:
            inpaintingClip.append([rows.img_id, rows.Clipscore])
    for index, rows in data.iterrows():
        if rows.SSIM_Inp >= thresholdSSIM:
            inpaintingssim.append([rows.img_id, rows.SSIM_Inp])
        if rows.SSIM_Erase >= thresholdSSIM:
            erasedata.append([rows.img_id, rows.SSIM_Erase])
        if rows.SSIM_Noise >= thresholdSSIM:
            noisedata.append([rows.img_id, rows.SSIM_Noise])

    print("number valid aug seed--clip",len(inpaintingClip))
    print("number valid aug seed--ssim", len(inpaintingssim))
    print("number valid aug seed erase", len(erasedata))
    print("number valid aug seed noise",len(noisedata))

    return inpaintingClip, inpaintingssim, erasedata, noisedata

def deleteLeadingZeros(inputString):
   inputString=inputString.replace('.jpg', '')
   inputString=str(inputString)
   regexPattern = "^0+(?!$)"
   outputString = r.sub(regexPattern, "", inputString)
   return outputString
def keep_valid_seeds(validseeds, dir_aug):
    # print("valid seeds:\n",validseeds)
    seeds = []
    removed=[]
    for val in validseeds:
        # print("inp clip",val[0])
        id= r.findall(r'\d+', val[0])
        # print("first",id)
        ix = deleteLeadingZeros(id[0])

        seeds.append(int(ix))

        # path = os.path.join(dir_aug, str(val[0]))
    print("all seed added", len(seeds))
    for i in os.listdir(dir_aug):
        ix, _ = os.path.splitext(i)
        # print("first",id)
        ix = deleteLeadingZeros(ix)
        # print("second", i)
        if int(ix) not in seeds:
        # if i not in seeds:
            os.remove(os.path.join(dir_aug, i))
        #     print('removed',os.path.join(dir_aug, i))
            # removed.append(id)
            removed.append(i)
    return removed


if __name__ == "__main__":
    dataset="coco_animal"
    if dataset=='cifar':
        categories = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        aug_dir_inpainting = "/home/sondess/sm2672/DataAugmentation_pipline/data/cifar/Augmented/Inpainting/"
        aug_dir_noise="/home/sondess/sm2672/DataAugmentation_pipline/data/cifar/Augmented/Noise/"
        aug_dir_erase="/home/sondess/sm2672/DataAugmentation_pipline/data/cifar/Augmented/Erasing/"
        inpaintingClip, inpaintingssim, erasedata, noisedata = select_valid_seed(
            '/home/sondess/sm2672/DataAugmentation_pipline/results/cifar_aug_set_validity.csv', 0.8, 0.6)
    if dataset == 'leaves':
        categories = ["Black_rot", "Esca_(Black_Measles)", "Healthy", "Leaf_blight_(Isariopsis_Leaf_Spot)"]
        aug_dir_inpainting = "/home/sondess/sm2672/DataAugmentation_pipline/data/Grapes/Augmented/Inpaintingknw"
        inpaintingClip, inpaintingssim, erasedata, noisedata = select_valid_seed(
            '/home/sondess/sm2672/DataAugmentation_pipline/results/cifar_aug_set_validity.csv', 0.8, 0.6)
    if dataset == 'coco_animal':
        categories = ['horse','sheep','zebra',"bear","bird",'cat','cow','dog','elephant','giraffe']#
        aug_dir_inpainting = "/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Inpainting"
        aug_dir_noise = "/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Noise/"
        aug_dir_erase = "/home/sondess/sm2672/DataAugmentation_pipline/data/coco/Augmented/Erasing/"
        inpaintingClip, inpaintingssim, erasedata, noisedata = select_valid_seed(
            '/home/sondess/sm2672/DataAugmentation_pipline/results/COCO_ANIMAL_ALL_validity.csv', 0.8, 0.6)

    total=0
    totalerase=0
    totalnoise=0
    for c in categories:
        pathinp = os.path.join(aug_dir_inpainting , c)
        pathnois = os.path.join(aug_dir_noise, c)
        pather = os.path.join(aug_dir_erase, c)
        # print("noise data", len(noisedata))
        # rnoise=keep_valid_seeds(noisedata, pathnois)

        rerase = keep_valid_seeds(erasedata, pather)
        # rinp = keep_valid_seeds(inpaintingClip, pathinp)
        # total=total+len(rinp)
        totalerase = totalerase + len(rerase)
        # totalnoise= totalnoise + len(rnoise)
    # print("removed noise", totalnoise)
    print("removed erase", totalerase)
    # print("removed inpainting",total)
