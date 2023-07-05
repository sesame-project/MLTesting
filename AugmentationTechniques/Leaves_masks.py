import cv2
import numpy as np
def masking_leaves(image, image_path):
    # Load the image
    # img1= cv2.imread(image_path)
    img=image
    if len(img.shape) != 3:
        print("shape")
        print(img.shape)
        # return 0
    else:
        # img = cv2.imread('./data/leaves/O_8.jpeg', cv2.IMREAD_UNCHANGED)
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        tmp = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        b, g, r = cv2.split(img)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        # cv2.imshow('Output', alpha)
        # cv2.imwrite("test.png", alpha)
        # cv2.waitKey(0)
        return alpha


""" wored with exceptions
   
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 
    # # Threshold the image to create a binary mask
    # _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 
    # # Find contours in the binary image
    # contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 
    # # Find the largest contour (i.e., the center object)
    # largest_contour = max(contours, key=cv2.contourArea)
    # 
    # # Create a mask for the center object
    # mask = np.zeros_like(gray)
    # cv2.drawContours(mask, [largest_contour], 0, 255, -1)
    # 
    # # Invert the mask
    # mask_inv = cv2.bitwise_not(mask)
    # 
    # # Apply the mask to the white background
    # background = np.full_like(img, (255, 255, 255))
    # result = cv2.bitwise_and(background, background, mask=mask_inv)
    # result += cv2.bitwise_and(img, img, mask=mask)
    return result
    """

