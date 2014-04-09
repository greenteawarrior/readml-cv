'''
Created on Apr 1, 2014

@author: dchen

extract waffles from the flickr imageset that jwei downloaded
using hue thresholding followed by histogram backprojection 

after segmenting the waffles and normalizing them to create a dataset, we'll take another approach by 
running keypoint identification followed by kmeans clustering
followed by logsitic regression to identify something as waffle 
or non waffle

^the second paragraph is pointless but its a ML exercise
'''
import cv2.cv as cv
import cv2
import os
import numpy as np
from matplotlib import pyplot as plt
    
def display_image_and_wait(image):
    cv2.imshow('dst',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def print_rgb_hist(img, mask):
    color = ('b','g','r')
    for i,col in enumerate(color):
        histr = cv2.calcHist([img],[i],mask,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
    plt.show()
    
def compare_hist(img1,mask1,img2,mask2):
    color = ('b','g','r')
    for i,col in enumerate(color):
        h1 = cv2.calcHist([img1],[i],mask1,[256],[0,256])
        h2 = cv2.calcHist([img2],[i],mask2,[256],[0,256])
    print cv2.compareHist(h1,h2,CV_COMP_CORREL)

def crop_waffle(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    greyscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    lower_yellow = np.array([0,50,50])
    upper_yellow = np.array([70,255,255])
    mask = cv2.inRange(hsv, np.uint8(lower_yellow), np.uint8(upper_yellow))
    kernel = np.ones((9,9),np.uint8)
    closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    masked_img = cv2.bitwise_and(greyscale,greyscale,mask = closed_mask)
    [contours,hiearchy] = cv2.findContours(masked_img,cv.CV_RETR_EXTERNAL,cv.CV_CHAIN_APPROX_SIMPLE)
    #now find the largest contour
    max_area = 0
    max_contour = None
    for c in contours:
        #we change datatypes from numpy arrays to cv arrays and back because contour area only takes cv arrays.
        c = cv.fromarray(c)
        if cv.ContourArea(c) > max_area:
            max_contour = c
            max_area = cv.ContourArea(c)
    max_contour = np.asarray(max_contour)
    shape = img.shape
    largest_blob_mask = np.zeros((shape[0],shape[1],1),np.uint8)
    cv2.fillPoly(largest_blob_mask, pts =[max_contour], color=(255,255,255))
    # print_rgb_hist(img,largest_blob_mask)
    return cv2.bitwise_and(img,img, mask= largest_blob_mask)

def main():
    waffle_folder_name = os.getcwd() + '/waffle_images'
    new_img_dir = os.getcwd() + '/modified_waffle_images'
    
    if not os.path.isdir(waffle_folder_name):
        os.makedirs(waffle_folder_name)
    if not os.path.isdir(new_img_dir):
        os.makedirs(new_img_dir)
        
    for index, pic in enumerate(os.listdir(waffle_folder_name)):
        loaded_pic = cv2.imread(waffle_folder_name + '/' + pic) 
        print waffle_folder_name + '/' + pic
        print type(loaded_pic)
        print 'reading: ' + waffle_folder_name + '/' + pic
        new_img_name = 'waffle_pic_'+ str(index) + '.jpg'
        cropped_pic = crop_waffle(loaded_pic)
        display_image_and_wait(cropped_pic)
        cv2.imwrite(new_img_dir + '/' + new_img_name,cropped_pic)

def compare_hist_tester():

    waffle_folder_name = os.getcwd() + '/waffle_images'
    new_img_dir = os.getcwd() + '/modified_waffle_images'    
    
    img1 = cv2.imread(waffle_folder_name+"/w45.jpeg")
    img2 = cv2.imread(waffle_folder_name+"/w5.jpeg")
    mask1 = crop_waffle(img1)
    mask2 = crop_waffle(img2)
    print mask1.dtype
    print mask2.dtype
    compare_hist(img1,mask1,img2,mask2)

compare_hist_tester()

