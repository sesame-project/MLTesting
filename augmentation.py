from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import re
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from matplotlib import pyplot as plt
# from stablediffusion.IMAGIC import *
# from wand.image import Image
from utils import *

from Augmentation_techniques.Leaves_masks import masking_leaves
import random as ran
from matplotlib import cm
import skimage.io as io
import urllib.request
import urllib
import skimage.color
import skimage.filters
import cv2
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
import os
from skimage import color
from skimage import io
from PIL import Image
import numpy as np
from Data_processing_pipline import Datapipline
from datetime import datetime
from Augmentation_techniques.Masking import *
from Augmentation_techniques.InpaintingDifussionModel import *
from Realism_measures.SSIM import *
from Realism_measures.FID import *
from Augmentation_techniques.Caption_Enrichement_NLP import *

nlp = spacy.load("en_core_web_sm")
directory_aug_data='./data/Grapes/'
def get_caption(label,category):
    Initial_caption = category[label]
    Initial_caption = Initial_caption
    return Initial_caption

def get_scores(ini_path,inpaint_path,height, width):
    load_images = lambda x: np.asarray(Image.open(x).resize((height, width)))
    # Helper functions to convert to Tensors
    tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)
    img_ini = load_images(ini_path)
    Aug_Img = load_images(inpaint_path)
    #######################
    _img_ini = image_channel(img_ini)
    _Aug_Img = image_channel(Aug_Img)
    #######################
    _img = tensorify(_img_ini)
    _img_aug = tensorify(_Aug_Img)
    img = Image.open(ini_path)
    img_initial = rescale_image(img)
    aug_img = rescale_image(Aug_Img)
    # print("the high and width \n")
    # print(img_initial.shape, aug_img.shape)
    fid_inpainting = FID(img_initial, aug_img)
    #  higher score indicates a lower-quality image and the relationship may be linear.
    print('Augmented vs original Image FID Score:: %.3f' % fid_inpainting)


    _img = _img
    _imgP = _img_aug
    # print("the SSIM high and width \n")
    # print(_img.shape, _imgP.shape)
    SSIM_inpaint = ssim(_img, _imgP, val_range=255)
    print("Augmented vs original Image SSIM Score:", SSIM_inpaint.item())
    return SSIM_inpaint.item(), fid_inpainting,
def augment_leaves_imgs(seed_size, initialset, approach, directory_aug_data, tax, categories):
    scores= []
    images= []
    subcat = 'grape leaves'
    augmented_path_inpaint = "./data/Grapes/Augmented/Inpainting/"
    augmented_path_erase = "./data/Grapes/Augmented/Erasing/"
    augmented_path_noise = "./data/Grapes/Augmented/Noise/"
    augmented_path_mask = "./data/Grapes/Augmented/Mask/"
    if approach =='Inpainting':
        id=3000
        ran.shuffle(initialset)
        for features, label in initialset[:seed_size]:
                id+=1
                print(id)
                ini_path = os.path.join(directory_aug_data,str(id) + ".jpg")
                height, width, _ = features.shape
                img_ini=features
                # print("feature shap",features.shape) is (256, 256)
                cv2.imwrite(ini_path, img_ini)
                mask=masking_leaves(img_ini, ini_path)
                if mask is not None:
                    plt.imshow(mask)
                    # print("Mask Type", type(mask))
                    mask_path = os.path.join(augmented_path_mask, str(id) + ".jpg")
                    cv2.imwrite(mask_path, mask)
                    print("saved mask")
                    Initial_caption = get_caption(label, categories)
                    print("Initial caption", Initial_caption)
                    caption=Initial_caption
                    rep = 0
                    for cat in categories:
                        if cat != caption:
                            print(cat)
                            current_path = os.path.join(augmented_path_inpaint, cat)
                            if not os.path.exists(current_path):
                                os.mkdir(current_path)
                            aug_caption_Category = Caption_category(Initial_caption, cat, subcat)
                            Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
                            Aug_Img = Aug_Imgs[0]
                            plt.imshow(Aug_Img)
                            plt.axis('off')
                            inpaint_path = os.path.join(current_path, str(rep)+str(id)+ ".jpg")
                            print(inpaint_path)
                        # print("TYPE Aug", type(Aug_Img))#PIL image
                            Aug_Img_save=numpy.array(Aug_Img)
                            Aug_Img_save= cv2.resize(Aug_Img_save, (height, width))
                            cv2.imwrite(inpaint_path, Aug_Img_save)
                            clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
                            print("the threshold score", clip_score.item())

                            img = Image.open(ini_path)
                            x = transforms.ToTensor()(img)
                            random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                            aug = random_erase(x).permute(1, -1, 0)
                            plt.imshow(np.squeeze(aug))
                            plt.axis('off')
                            current_erasepath = os.path.join(augmented_path_erase, cat)
                            if not os.path.exists(current_erasepath):
                                os.mkdir(current_erasepath)
                                print(current_erasepath)
                            erase_pa = str(rep) + str(id)+".jpg"
                            erase_path = os.path.join(current_erasepath, erase_pa)
                            plt.savefig(erase_path)

                            im = cv2.imread(ini_path, 0)
                            mean = 0
                            std = 1
                            gaus_noise = np.random.normal(mean, std, im.shape)
                            image = im.astype("int16")
                            noise_img = image + gaus_noise
                            noise_img = cv2.resize(noise_img, (height, width))
                            plt.imshow(noise_img)
                            current_noisepath = os.path.join(augmented_path_noise, cat)
                            if not os.path.exists(current_noisepath):
                                os.mkdir(current_noisepath)
                            noise_pa = (str(rep) + str(id)+".jpg")
                            noise_path = os.path.join(current_noisepath, noise_pa)
                            plt.imsave(noise_path, noise_img)
                            # cv2.imwrite(noise_path, noise_img)
                            SSIM_N, FID_Noise = get_scores(ini_path, noise_path,height, width)
                            SSIM_inpaint, FID_inpainting=get_scores(ini_path, inpaint_path,height, width)
                            SSIM_E, FID_Erase=get_scores(ini_path, erase_path,height, width)
                            line = [str(id), SSIM_inpaint, SSIM_E, FID_inpainting, FID_Erase, SSIM_N, FID_Noise, clip_score.item()]
                            scores.append(line)
                            images.append([str(id), Initial_caption, aug_caption_Category, ini_path, inpaint_path, erase_path, noise_path, label, cat])

                else:
                    continue


    return scores, images



def augment_from_folder(seed_size,approach, directory_aug_data, folder_dir, categories,ImgsIds, categories_keys):
    scores = []
    images = []
    inpainting_testing_data=[]
    erasing_testing_data = []
    noise_testing_data = []
    testing_data=[]
    Captions=get_img_caption(ImgsIds)
    augmented_path_inpaint="./data/coco/Augmented/Inpainting/"
    augmented_path_erase = "./data/coco/Augmented/Erasing/"
    augmented_path_noise = "./data/coco/Augmented/Noise/"
    if approach == 'Inpainting':
        for c in categories:
            print('label',c)
            label=c
            path = os.path.join(folder_dir, c)
            class_num = categories.index(c)
            count = 0
            # while count <=1:
            for i in os.listdir(path):
                    count = count + 1
                    print("file path",i)
                    ini_path=os.path.join(path, i)
                    try:
                        img_array = cv2.imread(os.path.join(path, i))
                        # img_array=cv2.resize(img_array,(128,128))
                        testing_data.append([img_array, class_num])
                    except Exception as e:
                        pass

                    height, width, _ = img_array.shape
                    img_ini = img_array
                    mask_path = os.path.join(directory_aug_data, "Masking/" + i)
                    Masking = Create_Mask(img_ini, mask_path)
                    W_mask = Masking.get_mask()
                    mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
                    plt.imsave(mask_path, mask)
                    print("saved mask")
                    id =int(deleteLeadingZeros(i))
                    # print("id",id)
                    if id in Captions.keys():
                        Initial_caption = Captions[id]["caption"]
                        print("Initial caption", Initial_caption)
                    rep = 0
                    for cat in categories:
                        if cat != label:
                            print(cat)
                            current_path=os.path.join(augmented_path_inpaint,cat)
                            if not os.path.exists(current_path):
                                os.mkdir(current_path)
                            if str(label) in str(Initial_caption).lower():
                                aug_caption= Change_caption(nlp,Initial_caption,label,"animal",cat)
                                print("aug:: ",aug_caption)
                            else:
                                aug_caption=cat
                                print("cap:: ", aug_caption)
                            class_nbr = categories.index(cat)
                            Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption)
                            Aug_Img = Aug_Imgs[0]
                            plt.imshow(Aug_Img)
                            plt.axis('off')
                            inpaint = (str(rep) + str(i))
                            inpaint_path = os.path.join(current_path, inpaint)
                            # print("TYPE Aug", type(Aug_Img))£#PIL image
                            Aug_Img_save = numpy.array(Aug_Img)
                            Aug_Img_save = cv2.resize(Aug_Img_save, (height, width))
                            # img = cv2.resize(img, (IMG_SIZ, IMG_SIZ), 3)
                            cv2.imwrite(inpaint_path, Aug_Img_save)
                            plt.imsave(inpaint_path, Aug_Img_save)
                            inpainting_testing_data.append([Aug_Img_save, class_nbr])
                            print("aug saved")
                            # rep=rep+1
                            clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption)
                            print("the threshold score", clip_score.item())

                            # erase#######
                            img = Image.open(ini_path)
                            x = transforms.ToTensor()(img)
                            random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                            aug = random_erase(x).permute(1, -1, 0)
                            plt.imshow(np.squeeze(aug))
                            # clip_erase = p_get_clip_score(np.squeeze(aug), Initial_caption, label)
                            # print("the erase clip score", clip_erase.item())
                            plt.axis('off')
                            current_erasepath = os.path.join(augmented_path_erase, cat)
                            if not os.path.exists(current_erasepath):
                                os.mkdir(current_erasepath)
                            erase_pa= (str(rep) + str(i))
                            erase_path = os.path.join(current_erasepath, erase_pa)
                            plt.savefig(erase_path)
                            erasing_testing_data.append([aug, class_num])
                            current_noisepath = os.path.join(augmented_path_noise, cat)
                            if not os.path.exists(current_noisepath):
                                os.mkdir(current_noisepath)
                            noise_pa = (str(rep) + str(i))
                            noise_path = os.path.join(current_noisepath, noise_pa)
                            im = cv2.imread(ini_path, 0)
                            mean = 0
                            std = 1
                            gaus_noise = np.random.normal(mean, std, im.shape)
                            image = im.astype("int16")
                            noise_img = image + gaus_noise
                            # noise_img = np.array(noise_img)
                            # clip_noise = p_get_clip_score(noise_img, Initial_caption, label)
                            # print("the noise clip score", clip_noise.item())
                            noise_img = cv2.resize(noise_img, (height, width))
                            plt.imshow(noise_img)
                            plt.imsave(noise_path, noise_img)
                            noise_testing_data.append([noise_img, class_num])
                            SSIM_N, FID_Noise = get_scores(ini_path, noise_path, height, width)
                            SSIM_inpaint, FID_inpainting = get_scores(ini_path, inpaint_path, height, width)
                            SSIM_E, FID_Erase = get_scores(ini_path, erase_path, height, width)
                            line = [str(i), SSIM_inpaint, SSIM_E, FID_inpainting, FID_Erase, SSIM_N, FID_Noise, clip_score.item()]
                            scores.append(line)
                            images.append([str(id), Initial_caption, aug_caption,ini_path,inpaint_path, erase_path,noise_path,label,cat])



    return scores, images,inpainting_testing_data,erasing_testing_data,noise_testing_data


'''
augment_coco_imgs works perfectly and rely on mask rnn to create masks !
'''
def augment_coco_imgs(seed_size, approach, directory_aug_data):
    results = []
    images = []
    nlp = spacy.load("en_core_web_sm")
    datapath = "./data/Initial/annotations/instances_train2017.json"
    KeyObjFile = "./data/Initial/annotations/person_keypoints_train2017.json"
    CaptionFile = "./data/Initial/annotations/captions_train2017.json"
    cat = 'dog'
    subcat = 'person'
    rep = 'panda'
    filterClasses3 = [cat, subcat]
    pre = Datapipline(datapath, KeyObjFile, CaptionFile)
    maincategories, subcategories = pre.load_cat_info()
    Initial_seed_Set, Initial_seed_Ids = pre.select_subCatg(filterClasses3)
    next_pix = Initial_seed_Set
    ran.shuffle(next_pix)
    coco_kps = pre.KObj
    coco_caps = pre.Caps
    # Load_images_with keypoints objects and captions
    for i, img_path in enumerate(next_pix[0:seed_size]):
        img = pre.coco.loadImgs(img_path)[0]
        id = str(img['id'])
        # if id in ['518951','77709']:
        #     continue
        # else:
        if len(img.shape) != 3:
            print("shape")
            print(img.shape)
            continue
        else:
            I = io.imread(img['coco_url'])

            ini_path = os.path.join(directory_aug_data, "Initial2/I_" + id + ".jpeg")
            mask_path = os.path.join(directory_aug_data, "masks/M_" + id + ".jpeg")
            erase_path = os.path.join(directory_aug_data, "erase/E_" + id + ".jpeg")
            noise_path = os.path.join(directory_aug_data, "noise/N_" + id + ".jpeg")
            img_pil = PIL.Image.open(urllib.request.urlopen(img['coco_url']))
            annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=Initial_seed_Ids, iscrowd=None)
            anns = coco_kps.loadAnns(annIds)

            CaptionIds = coco_caps.getAnnIds(imgIds=img['id'])
            caption = coco_caps.loadAnns(CaptionIds)
            if cat not in str(caption[0]['caption']).lower():
                continue
            else:
                Initial_caption = caption[0]['caption']
                plt.imsave(ini_path, I)
                aug_caption_Category = Change_caption(nlp, Initial_caption, cat, subcat, rep)
                """
                    Augmentation process
                """
                    # 1/ create masque
                img = Image.open(ini_path)

                Masking = Create_Mask(I, mask_path)
                W_mask = Masking.get_mask()
                if type(W_mask)==np.ndarray :

                    mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
                    plt.imsave(mask_path, mask)
                    print("saved mask")
                    Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
                    Aug_Img = Aug_Imgs[0]
                    plt.imshow(Aug_Img)
                    plt.axis('off')
                    inpaint_path = os.path.join(directory_aug_data,"Inpaint/In_" + id + ".jpeg")
                    plt.savefig(inpaint_path)
                    clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
                    if clip_score < 70:
                        print("the score:", clip_score.item())
                        continue
                    else:
                        print("the score", clip_score.item())
                        plt.savefig(inpaint_path)
                        img = Image.open(ini_path)
                        x = transforms.ToTensor()(img)
                        random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                        aug = random_erase(x).permute(1, -1, 0)
                        plt.imshow(np.squeeze(aug))
                        plt.axis('off')
                        plt.savefig(erase_path)
                        # Gaussian Noise

                        im = cv2.imread(ini_path, 0)
                        mean = 0
                        std = 1
                        gaus_noise = np.random.normal(mean, std, im.shape)
                        image = im.astype("int16")
                        noise_img = image + gaus_noise
                        plt.imshow(noise_img)
                        plt.imsave(noise_path, noise_img)
                        SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise = get_scores(ini_path,
                                                                                                        inpaint_path,
                                                                                                        erase_path,
                                                                                                        noise_path)
                        line = [str(id), SSIM_inpaint, SSIM_E, SSIM_N, FID_inpainting, FID_Erase, FID_Noise,
                                clip_score.item()]
                        results.append(line)
                        images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category,
                                       "original": ini_path, "Inpainting": inpaint_path, "Erase": erase_path,
                                       "Noise": noise_path})
                else:
                    continue

    return results,images




def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id'] == classID:
            return cats[i]['name']
    return "None"
"""
'''
augment_ALLcoco relies on mask from coco annotation file
'''

"""
def augment_ALLcoco(seed_size,directory_aug_data, cat):
    results = []
    images = []
    nlp = spacy.load("en_core_web_sm")
    datapath = "./data/Initial/annotations/instances_train2017.json"
    KeyObjFile = "./data/Initial/annotations/person_keypoints_train2017.json"
    CaptionFile = "./data/Initial/annotations/captions_train2017.json"
    # cat = 'dog'
    subcat = 'person'
    rep = 'men'
    filterClasses = [cat]
    pre = Datapipline(datapath, KeyObjFile, CaptionFile)
    maincategories, subcategories = pre.load_cat_info()
    Initial_seed_Set, Initial_seed_Ids = pre.select_subCatg(filterClasses)
    next_pix = Initial_seed_Set
    ran.shuffle(next_pix)
    coco_kps = pre.KObj
    coco_caps = pre.Caps
    coco = pre.coco

    catIDs = coco.getCatIds()
    cats = coco.loadCats(catIDs)
    # Fetch class IDs only corresponding to the filterClasses
    catIds = coco.getCatIds(catNms=filterClasses)
    # Get all images containing the above Category IDs
    imgIds = coco.getImgIds(catIds=catIds)
    print("Number of images containing all the  classes:", len(imgIds))
    ran.shuffle(imgIds)
    categories=['person','people','kids','kid','child', 'children','girl','boy','woman','women','man','men']
    # Load_images_with keypoints objects and captions
    for i, img_path in enumerate(imgIds[0:seed_size]):
        img = coco.loadImgs(img_path)[0]
        id = str(img['id'])
        I = io.imread(img['coco_url'])
        # if id in ['518951','77709']:
        #     continue
        # else:
        if len(I.shape) != 3:
            print("shape")
            print(I.shape)
            continue
        else:

                        height, width, _ = I.shape
                        ini_path = os.path.join(directory_aug_data, "coco/" + id + ".jpeg")
                        mask_path = os.path.join(directory_aug_data, "coco/M_" + id + ".jpeg")
                        secondary=os.path.join(directory_aug_data, "coco/Mask_" + id + ".jpeg")

                        img_pil = PIL.Image.open(urllib.request.urlopen(img['coco_url']))
                        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=0)
                        anns = coco.loadAnns(annIds)
                        captionIds = coco_caps.getAnnIds(imgIds=img['id'])
                        caption = coco_caps.loadAnns(captionIds)
                        label = caption[0]['caption']

            # if cat not in str(caption[0]['caption']).lower():
            # if label not in categories:
            #     continue
            # else:
                        mask = np.zeros((img['height'], img['width']))
                        for i in range(len(anns)):
                            className = getClassName(anns[i]['category_id'], cats)
                            pixel_value = filterClasses.index(className) + 1
                            mask = np.maximum(coco.annToMask(anns[i]) * pixel_value, mask)

                        plt.imsave(secondary, mask)
                        Imask = io.imread(secondary)
                        gray_image = skimage.color.rgb2gray(Imask)
                        # blur the image to denoise
                        blurred_image = skimage.filters.gaussian(gray_image, sigma=1.0)
                        plt.imshow(blurred_image, cmap="gray")
                        plt.axis('off')
                        # plt.imshow(mask)
                        plt.imsave(mask_path, blurred_image, cmap="gray")
                        print("saved mask")
                        os.remove(secondary)
                        Initial_caption = caption[0]['caption']
                        plt.imsave(ini_path, I)
                        aug_caption_Category = Change_caption(nlp, Initial_caption, cat, subcat, rep)
                        print("new label \n")
                        print(aug_caption_Category)
                        """
                            Augmentation process
                        """
                            # 1/ create masque
                        # img = Image.open(ini_path)
                        for rep in range(10):
                            erase_path = os.path.join(directory_aug_data, "coco/E_" + str(id) + "_" + str(rep) + ".jpeg")
                            noise_path = os.path.join(directory_aug_data, "coco/N_" + str(id) + "_" + str(rep) +".jpeg")
                            inpaint_path = os.path.join(directory_aug_data, "coco/In_" + str(id) + "_" + str(rep) + ".jpeg")
                            Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
                            Aug_Img = Aug_Imgs[0]
                            plt.imshow(Aug_Img)
                            plt.axis('off')
                            Aug_Img_save = numpy.array(Aug_Img)
                            Aug_Img_save == cv2.resize(Aug_Img_save, (height, width))
                            cv2.imwrite(inpaint_path, Aug_Img_save)
                            clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
                            print("the threshold score", clip_score.item())
                            # plt.imsave(inpaint_path, Aug_Img)
                            # erase#######
                            img = Image.open(ini_path)
                            x = transforms.ToTensor()(img)
                            random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                            aug = random_erase(x).permute(1, -1, 0)
                            plt.imshow(np.squeeze(aug))
                            plt.axis('off')
                            erase_path = os.path.join(directory_aug_data, "coco/E_" + str(id) + "_" + str(rep) + ".jpeg")
                            plt.savefig(erase_path)
                            # print("sq type", type(np.squeeze(aug)))
                            # print("type", type(aug))
                            # cv2.imwrite(erase_path, np.squeeze(aug))
                            noise_path = os.path.join(directory_aug_data, "coco/N_" + str(id) + "_" + str(rep) + ".jpeg")
                            im = cv2.imread(ini_path, 0)
                            mean = 0
                            std = 1
                            gaus_noise = np.random.normal(mean, std, im.shape)
                            image = im.astype("int16")
                            noise_img = image + gaus_noise
                            noise_img = cv2.resize(noise_img, (height, width))
                            plt.imshow(noise_img)
                            plt.imsave(noise_path, noise_img)
                            # cv2.imwrite(noise_path, noise_img)
                            SSIM_N, FID_Noise = get_scores(ini_path, noise_path, height, width)
                            SSIM_inpaint, FID_inpainting = get_scores(ini_path, inpaint_path, height, width)
                            SSIM_E, FID_Erase = get_scores(ini_path, erase_path, height, width)
                            line = [str(id), SSIM_inpaint, SSIM_E, FID_inpainting, FID_Erase, SSIM_N, FID_Noise,
                                    clip_score.item()]
                            results.append(line)
                            images.append({"id": str(id), "caption": Initial_caption, "Aug_caption": aug_caption_Category,
                                           "original": ini_path, "Inpainting": inpaint_path, "Erase": erase_path})



    return results,images
def augment_CIFAR_imgs(seed_size, X_test, y_test, approach, directory_aug_data, categories):
    scores = []
    images = []
    inpainting_testing_data = []
    erasing_testing_data = []
    noise_testing_data = []
    testing_data = []
    augmented_path_inpaint = "./data/cifar/Augmented/Inpainting/"
    augmented_path_erase = "./data/cifar/Augmented/Erasing/"
    augmented_path_noise = "./data/cifar/Augmented/Noise/"
    augmented_path_mask = "./data/cifar/Augmented/Mask/"
    if approach == 'Inpainting':
        id = 100
        for features, label in zip(X_test[:seed_size], y_test[:seed_size]):
                id += 1
                print(id)
                ini_path = os.path.join(directory_aug_data, str(id) + ".jpg")
                height, width, _ = features.shape
                img_ini = features
                cv2.imwrite(ini_path, img_ini)
                print("label", np.argmax(label))
                plt.imshow(features)
                plt.savefig(ini_path)
                cv2.imwrite(ini_path, img_ini)
                mask_path = os.path.join(augmented_path_mask, str(id) + ".jpg")
                Masking = Create_Mask(img_ini, mask_path)
                W_mask = Masking.get_mask()
                mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
                plt.imshow(mask)
                cv2.imwrite(mask_path, mask)
                print("saved mask")
            # if mask is not None:


                Initial_caption = get_caption(label[0], categories)
                print("Initial caption", Initial_caption)
                caption = Initial_caption
                rep = 0
                subcat = ''
                for cat in categories:
                    if cat != caption:
                        print(cat)
                        current_path = os.path.join(augmented_path_inpaint, cat)
                        if not os.path.exists(current_path):
                            os.mkdir(current_path)
                        aug_caption_Category = Caption_category(Initial_caption, cat, subcat)
                        Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
                        Aug_Img = Aug_Imgs[0]
                        plt.imshow(Aug_Img)
                        plt.axis('off')
                        inpaint_path = os.path.join(current_path, str(rep) + str(id) + ".jpg")
                        print(inpaint_path)
                        # print("TYPE Aug", type(Aug_Img))#PIL image
                        Aug_Img_save = numpy.array(Aug_Img)
                        Aug_Img_save = cv2.resize(Aug_Img_save, (height, width))
                        cv2.imwrite(inpaint_path, Aug_Img_save)
                        clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
                        print("the threshold score", clip_score.item())

                        img = Image.open(ini_path)
                        x = transforms.ToTensor()(img)
                        random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
                        aug = random_erase(x).permute(1, -1, 0)
                        plt.imshow(np.squeeze(aug))
                        plt.axis('off')
                        current_erasepath = os.path.join(augmented_path_erase, cat)
                        if not os.path.exists(current_erasepath):
                            os.mkdir(current_erasepath)
                        erase_pa = str(rep) + str(id) + ".jpg"
                        erase_path = os.path.join(current_erasepath, erase_pa)
                        plt.savefig(erase_path)

                        im = cv2.imread(ini_path, 0)
                        mean = 0
                        std = 1
                        gaus_noise = np.random.normal(mean, std, im.shape)
                        image = im.astype("int16")
                        noise_img = image + gaus_noise
                        noise_img = cv2.resize(noise_img, (height, width))
                        plt.imshow(noise_img)
                        current_noisepath = os.path.join(augmented_path_noise, cat)
                        if not os.path.exists(current_noisepath):
                            os.mkdir(current_noisepath)
                        noise_pa = (str(rep) + str(id) + ".jpg")
                        noise_path = os.path.join(current_noisepath, noise_pa)
                        plt.imsave(noise_path, noise_img)
                        # cv2.imwrite(noise_path, noise_img)
                        SSIM_N, FID_Noise = get_scores(ini_path, noise_path, height, width)
                        SSIM_inpaint, FID_inpainting = get_scores(ini_path, inpaint_path, height, width)
                        SSIM_E, FID_Erase = get_scores(ini_path, erase_path, height, width)
                        line = [str(id), SSIM_inpaint, SSIM_E, FID_inpainting, FID_Erase, SSIM_N, FID_Noise,
                                clip_score.item()]
                        scores.append(line)
                        images.append(
                            [str(id), Initial_caption, aug_caption_Category, ini_path, inpaint_path, erase_path,
                             noise_path, label, cat])




    return scores, images

def augment_KNW_imgs(seed_size, X_test, y_test, approach, directory_aug_data, categories,data,id):
    scores = []
    images = []
    inpainting_testing_data = []
    erasing_testing_data = []
    noise_testing_data = []
    testing_data = []

    augmented_path_inpaint = "./data/"+data+"/Augmented/Inpaintingknw/"
    augmented_path_erase = "./data/"+data+"/Augmented/Erasing/"
    augmented_path_noise = "./data/"+data+"/Augmented/Noise/"
    augmented_path_mask = "./data/"+data+"/Augmented/Mask/"
    if approach == 'Inpainting':
            for features, label in zip(X_test[:seed_size], y_test[:seed_size]):
                id += 1
                print(id)
                ini_path = os.path.join(directory_aug_data, str(id) + ".jpg")
                height, width, _ = features.shape
                img_ini = features
                cv2.imwrite(ini_path, img_ini)
                print("label", np.argmax(label))
                plt.imshow(features)
                plt.savefig(ini_path)
                cv2.imwrite(ini_path, img_ini)
                mask_path = os.path.join(augmented_path_mask, str(id) + ".jpg")
                Masking = Create_Mask(img_ini, mask_path)
                W_mask = Masking.get_mask()
                mask = cv2.cvtColor(W_mask, cv2.COLOR_BGR2RGB)
                plt.imshow(mask)
                cv2.imwrite(mask_path, mask)
                print("saved mask")
                # if mask is not None:

                Initial_caption = get_caption(np.argmax(label), categories)#(label[0], categories)
                print("Initial caption", Initial_caption)
                caption = Initial_caption
                rep = 0
                subcat = ''
                for cat in categories:
                    if cat != caption:
                        print(cat)
                        current_path = os.path.join(augmented_path_inpaint, cat)
                        if not os.path.exists(current_path):
                            os.mkdir(current_path)
                        aug_caption_Category = Caption_category(Initial_caption, cat, subcat)
                        Aug_Imgs = Inpainting(ini_path, mask_path, aug_caption_Category)
                        Aug_Img = Aug_Imgs[0]
                        plt.imshow(Aug_Img)
                        plt.axis('off')
                        inpaint_path = os.path.join(current_path, str(rep) + str(id) + ".jpg")
                        print(inpaint_path)
                        # print("TYPE Aug", type(Aug_Img))#PIL image
                        Aug_Img_save = numpy.array(Aug_Img)
                        Aug_Img_save = cv2.resize(Aug_Img_save, (height, width))
                        cv2.imwrite(inpaint_path, Aug_Img_save)
                        clip_score = p_get_clip_score(Aug_Img, Initial_caption, aug_caption_Category)
                        print("the threshold score", clip_score.item())
                        SSIM_inpaint, FID_inpainting = get_scores(ini_path, inpaint_path, height, width)
                        line = [str(id), SSIM_inpaint, FID_inpainting, clip_score.item()]
                        scores.append(line)





    return scores, images

