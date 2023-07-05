import os
import cv2
import numpy as np
import glob
import os
from  scipy import ndimage
import cv2
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
def create_separate_mask(path):
    # get all the masks
    # for mask_file in glob.glob(path):
        mask = cv2.imread(path, 1)
        # get masks labelled with different values
        label_im, nb_labels = ndimage.label(mask)

        for i in range(nb_labels):

            # create an array which size is same as the mask but filled with
            # values that we get from the label_im.
            # If there are three masks, then the pixels are labeled
            # as 1, 2 and 3.

            mask_compare = np.full(np.shape(label_im), i+1)

            # check equality test and have the value 1 on the location of each mask
            separate_mask = np.equal(label_im, mask_compare).astype(int)

            # replace 1 with 255 for visualization as rgb image

            separate_mask[separate_mask == 1] = 255
            base=os.path.basename(mask_file)

            # give new name to the masks

            file_name = os.path.splitext(base)[0]
            file_copy = os.path.join(path, file_name + "_" + str(i+1) +".png")
            cv2.imwrite(file_copy, separate_mask)
if __name__ == "__main__":
     path_to_frozen_inference_graph = '../models/frozen_inference_graph.pb'
     path_coco_model = '../models/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt'
     image_path = '../coco/s.jpeg'
     # Loading Mask RCNN
     net = cv2.dnn.readNetFromTensorflow(path_to_frozen_inference_graph,path_coco_model)
     colors = np.random.randint(125, 255, (80, 3))
     img = cv2.imread(image_path)
     img=cv2.resize(img,(650,550))
     height, width, _ = img.shape
     black_image = np.zeros((height, width, 3), np.uint8)
     black_image[:] = (0, 0, 0)
     blob = cv2.dnn.blobFromImage(img, swapRB=True)
     net.setInput(blob)
     boxes, masks = net.forward(["detection_out_final", "detection_masks"])
     detection_count = boxes.shape[2]
     for i in range(detection_count):
          box = boxes[0, 0, i]
          class_id = box[1]
          score = box[2]
          if score < 0.5:
              continue
          x = int(box[3] * width)
          y = int(box[4] * height)
          x2 = int(box[5] * width)
          y2 = int(box[6] * height)
          roi = black_image[y: y2, x: x2]
          roi_height, roi_width, _ = roi.shape
          mask = masks[i, int(class_id)]
          mask = cv2.resize(mask, (roi_width, roi_height))
          _, mask = cv2.threshold(mask, 0.5, 255, cv2.THRESH_BINARY)
          cv2.rectangle(img, (x, y), (x2, y2), (255, 0, 0), 3)
          contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
          color = colors[int(class_id)]
          for cnt in contours:
              cv2.fillPoly(roi, [cnt], (int(color[0]), int(color[0]), int(color[0])))
     cv2.imshow("Final",np.hstack([img,black_image]))
     # cv2.imshow("msk",(black_image))
     # cv2.imshow("m",(mask))
     cv2.imwrite("masks/mask3.png",black_image)

     cv2.imshow("Overlay_image",((0.6*black_image)+(0.4*img)).astype("uint8"))
     cv2.waitKey(0)
     create_separate_mask("../masks/mask3.png")
     print("saved")