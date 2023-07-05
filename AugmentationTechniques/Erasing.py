# ! conda install -c conda-forge timm
import timm
from PIL import Image
import numpy as np
import cv2
from timm.data.random_erasing import RandomErasing
from torchvision import transforms
from matplotlib import pyplot as plt
from DataAugmentation_pipline import *
input_image='obamatest6.jpg'
img = Image.open('/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/obamatest6.jpg')
x   = transforms.ToTensor()(img)
plt.imshow(x.permute(1, 2, 0))
# plt.show()
img = Image.open('/Users/sondessmissaoui/PycharmProjects/DataAugmentation_pipline/SSIM-PyTorch/obamatest6.jpg')
x   = transforms.ToTensor()(img)
plt.imshow(x.permute(1, 2, 0))
plt.show()
random_erase = RandomErasing(probability=1, mode='pixel', device='cpu')
aug=random_erase(x).permute(1, -1, 0)
plt.imshow(np.squeeze(aug))
plt.title("Erasing initial results")
plt.savefig("e.png", dpi=100)
plt.show()


