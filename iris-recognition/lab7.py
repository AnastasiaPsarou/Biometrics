import cv2
from os import listdir
from sklearn.metrics import classification_report 
import os
import numpy as np


for file in listdir('/home/anastasia/Έγγραφα/biometrics/lab7/RIDB-20220426T114609Z-001/RIDB'):
    filename = os.fsdecode(file)        
    image = cv2.imread('/home/anastasia/Έγγραφα/biometrics/lab7/RIDB-20220426T114609Z-001/RIDB/' + filename)
    
    #grayscale
    img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    #use floodfil algorithm (cv2.floodFill) from the point (0, 0)
    cv2.floodFill(img_gray, None, (0, 0), 255)
    
    #remember to work on the gray image copy:
    img_floodfill = img_gray.copy()
    
    #to obtain proper mask one can use the following code - READY
    h, w = img_floodfill.shape[:2]
    floodfill_mask = np.zeros((h+2, w+2), np.uint8)
    
    #dilate the floodfill result to cover pixels near the ROI (retina image) - use kernel of size 11 or 13 - READY
    dil_kernel = np.ones((13, 13),np.uint8)
    
    #finally, do the bitwiste_not and binarization with small threshold (e.g. 1) toobtain the proper mask    
    floodfill_mask = cv2.bitwise_not(img_floodfill)
    (T, floodfill_mask) = cv2.threshold(img_floodfill, 1, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Threshold", floodfill_mask)
    
    #enhance contrast of the grayscale image using CLAHE algorithm (clipLimit = 2.0, tileGridSize = (8,8))
    clahe = cv2.createCLAHE(2, (8,8))
    final_image = clahe.apply(img_gray) + 30
    
    #invert enhanced grayscale image (255 - img) and once again use the CLAHE algorithm
    final_image = 255 - final_image
    final_image = clahe.apply(final_image)
    
    #blur image with the gaussian kernel (size 7x7, sigma calculated from size)
    im_gauss = cv2.GaussianBlur(final_image, (7,7), 0)
    
    #do the adaptive thresholding with gaussian-weighted sum of the neighbourhood values (cv2.ADAPTIVE_THRESH_GAUSSIAN_C, C = 0)
    im_threshold = cv2.adaptiveThreshold(im_gauss, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    #combine threshold output with previously prepared mask (cv2.bitwise_and)
    img = cv2.bitwise_and(im_threshold, floodfill_mask)
    
    #blur image with the median filter (size 5x5) and invert values (cv2.bitwise_not)
    img = cv2.medianBlur(img, 5)
    img = cv2.bitwise_not(img)
    
    #fill the holes with morphological close operation - you can use kernel from previous dilation
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)))
    img = cv2.bitwise_not(img)
        
    #get objects stats and remove background object
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    sizes = stats[1:, -1]
    nb_components = nb_components - 1
    
    #remove smaller objects (area < 500)
    img2 = np.zeros((output.shape), np.uint8)
    for i in range(0, nb_components):
        if stats[i + 1, cv2.CC_STAT_AREA] >= 500:
            img2[i + 1] = 255
            
    cv2.imshow("result", img2)
    cv2.imshow("original", image)
    cv2.waitKey()
    
    #finally, get the skeleton of the preserved objects - you can use the following code
    skel_element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
    skel = np.zeros(img2.shape, np.uint8)
    while True:
        opened = cv2.morphologyEx(img2, cv2.MORPH_OPEN, skel_element)
        temp = cv2.subtract(img2, opened)
        eroded = cv2.erode(img2, skel_element)
        skel = cv2.bitwise_or(skel, temp)
        img2 = eroded.copy()
        
        if cv2.countNonZero(img2)==0:
            break
    
