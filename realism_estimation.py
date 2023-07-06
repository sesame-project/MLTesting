from Realism_measures.SSIM import *
from Realism_measures.FID import *
import argparse

import os
from datetime import datetime
def parse_arguments():
    """
    Parse command line argument and construct the DNN
    :return: a dictionary comprising the command-line arguments
    """



    text = 'Augmented Image Realism'

    # initiate the parser
    parser = argparse.ArgumentParser(description=text)


    parser.add_argument("-M", "--measure", help="measure to be used", choices=['SSIM','FID'])
    parser.add_argument("-DS", "--dataset", help="The path of the folder containing augmneted images")
    # parser.add_argument("-P", "--percentage", help="the percentage of TrKnw neurons to be deployed", type=float)

    args = parser.parse_args()

    return vars(args)
def realism_estimation(measure):
    load_images = lambda x: np.asarray(Image.open(x).resize((480, 640)))
    if measure=="SSIM":
        gauss_dis = gaussian(11, 1.5)
        print("Distribution: ", gauss_dis)
        print("Sum of Gauss Distribution:", torch.sum(gauss_dis))
        window = create_window(11, 3)
        print("Shape of gaussian window:", window.shape)
        # helper function to load images
        # load_images = lambda x: np.asarray(Image.open(x).resize((480, 640)))

        # Helper functions to convert to Tensors
        tensorify = lambda x: torch.Tensor(x.transpose((2, 0, 1))).unsqueeze(0).float().div(255.0)
        # The true reference Image
        img1 = load_images("/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/obamatest6.jpg")

        # The False image
        img2 = load_images("/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/zucktest3.jpeg")

        # The noised true image
        noise = np.random.randint(0, 255, (640, 480, 3)).astype(np.float32)
        noisy_img = img1 + noise

        print("True Image\n")
        display_imgs(img1)

        print("\nFalse Image\n")
        display_imgs(img2)

        print("\nNoised True Image\n")
        display_imgs(noisy_img)
        # cv2.waitKey(0)
        # Check SSIM score of True image vs False Image
        _img1 = tensorify(img1)
        _img2 = tensorify(img2)
        true_vs_false = ssim(_img1, _img2, val_range=255)
        print("True vs False Image SSIM Score:", true_vs_false)
        # Check SSIM score of True image vs Noised_true Image
        _img1 = tensorify(img1)
        _img2 = tensorify(noisy_img)
        true_vs_false = ssim(_img1, _img2, val_range=255)
        print("True vs Noisy True Image SSIM Score:", true_vs_false)
        # Check SSIM score of True image vs True Image
        _img1 = tensorify(img1)
        true_vs_false = ssim(_img1, _img1, val_range=255)
        print("True vs True Image SSIM Score:", true_vs_false)
    elif measure=="FID":
        # prepare the inception v3 model
        model = InceptionV3(include_top=False, pooling='avg', input_shape=(299,299,3))
        # define two fake collections of images
        # images1 = randint(0, 255, 10*32*32*3)
        images1 = load_images("/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/obamatest6.jpg")

        # The False image
        images2 = load_images( "/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/zucktest3.jpeg")

        # images1 = images1.reshape((10,32,32,3))
        # images2 = randint(0, 255, 10*32*32*3)
        # images2 = images2.reshape((10,32,32,3))
        print('Prepared', images1.shape, images2.shape)
        # convert integer to floating point values
        images1 = images1.astype('float32')
        images2 = images2.astype('float32')
        # resize images
        images1 = scale_images(images1, (299,299,3))
        images2 = scale_images(images2, (299,299,3))
        print('Scaled', images1.shape, images2.shape)
        # pre-process images
        images1 = preprocess_input(images1)
        images2 = preprocess_input(images2)
        # fid between images1 and images1
        fid = calculate_fid(model, images1, images1)
        print('FID (same): %.3f' % fid)
        # fid between images1 and images2
        fid = calculate_fid(model, images1, images2)
        print('FID (different): %.3f' % fid)
if __name__ == "__main__":
    args = parse_arguments()
    measure = args['measure'] if args['measure'] else 'FID'
    dataset = args['dataset'] if args['dataset'] else './'


    # startTime = time.time()

    results = realism_estimation(measure)


    # endTime = time.time()
    # elapsedTime = endTime - startTime
    # print("Elapsed Time = %s" % elapsedTime)