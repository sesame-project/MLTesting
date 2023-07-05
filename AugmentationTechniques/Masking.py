import cv2
import numpy as np
import argparse
import random
import time
import os
import skimage.io as io
import matplotlib.pyplot as plt
# Loading Mask RCNN
import cv2
from PIL import Image


class Create_Mask():
    def __init__(self, img, path):

        # self.net = cv2.dnn.readNetFromTensorflow("/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/models/frozen_inference_graph.pb",
        #                                     "/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

        self.net = cv2.dnn.readNetFromTensorflow(
            "/home/sondess/sm2672/DataAugmentation_pipline/models/frozen_inference_graph.pb",
            "/home/sondess/sm2672/DataAugmentation_pipline/models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")

        self.img = cv2.resize(img, (256, 256))
        # print("type", type(img))
        self.colors = np.random.randint(125, 255, (90, 3))
        self.height, self.width, _ = img.shape
        self.black_image = np.zeros((self.height, self.width, 3), np.uint8)
        self.black_image[:] = (0, 0, 0)
        self.path=path

    def get_mask(self):
        if len(self.img.shape) != 3:
            print(self.img.shape)
            return 0
        else:
            blob = cv2.dnn.blobFromImage(self.img, swapRB=True)
            self.net.setInput(blob)
            boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])
            detection_count = boxes.shape[2]

            mask=[]

            for i in range(detection_count):
                    box = boxes[0, 0, i]
                    class_id = box[1]
                    score = box[2]
                    # print(score)
                    if score < 0.3:
                        # print("continue")
                        continue

                    # Get box Coordinates
                    x = int(box[3] * self.width)
                    y = int(box[4] * self.height)
                    x2 = int(box[5] * self.width)
                    y2 = int(box[6] * self.height)
                    roi2 = self.img[y: y2, x: x2]
                    roi = self.black_image[y: y2, x: x2]
                    roi_height, roi_width, _ = roi.shape
                    # print("the height and width",roi_height, roi_width, _)
                    # Get the mask
                    mask = masks[i, int(class_id)]
                    mask = cv2.resize(mask, (roi_width, roi_height))
                    _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
                    print(type(mask))
                    cv2.rectangle(self.img, (x, y), (x2, y2), (255, 0, 0), 3)
                    # plt.imshow(cv2.cvtColor(roi2, cv2.COLOR_BGR2RGB))

                    # plt.show()
                    # Get mask coordinates
                    contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    color = self.colors[int(class_id)]
                    for cnt in contours:
                        cv2.fillPoly(roi, [cnt], (255, 255, 255))
                        int(color[0]), int(color[0]), int(color[0])
                    # plt.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
                    # plt.show()
                    # cv2.imwrite(self.path+"im222222.png", roi2)
                    plt.imshow(cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))
                    # plt.show()
                    # return masks

        return self.black_image #mask, self.black_image








if __name__ == "__main__":
    start_time = time.time()
    image_path="../data/initial2/O_524291.png"
    path='../data/initial2/'
    # img = cv2.imread(image_path)
    img = io.imread(image_path)

    Masking=Create_Mask(img,path)
    masks, W_mask = Masking.get_mask()

