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

# def find_vertical_lines(img):
#     """takes an rgb image, converts it to grayscale, then finds and returns a vertical lines mask"""
#     greyscale = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#     kernelx = cv2.getStructuringElement(cv2.MORPH_RECT,(1,5))
#     
#     dx = cv2.Sobel(greyscale,cv2.CV_16S,1,0)
#     dx = cv2.convertScaleAbs(dx)
#     cv2.normalize(dx,dx,0,255,cv2.NORM_MINMAX)
#     ret,close = cv2.threshold(dx,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#     close = cv2.morphologyEx(close,cv2.MORPH_DILATE,kernelx,iterations = 1)
#     
#     contour, hier = cv2.findContours(close,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contour:
#         x,y,w,h = cv2.boundingRect(cnt)
#         if h/w > 5:
#             cv2.drawContours(close,[cnt],0,255,-1)
#         else:
#             cv2.drawContours(close,[cnt],0,0,-1)
#     close = cv2.morphologyEx(close,cv2.MORPH_CLOSE,None,iterations = 2)
#     inv_mask = cv2.bitwise_not(close)
#     return cv2.bitwise_and(greyscale,greyscale, mask=inv_mask)
#     
#     
# def find_horizontal_lines(img):
#     pass
# 
# 
# def process(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     edges = cv2.Canny(gray,100,200)
#     
#     lines = cv2.HoughLines(edges,1,np.pi/180,100)
#     if lines != None:
#         for rho, theta in lines[0]:
#             a = np.cos(theta)
#             b = np.sin(theta)
#             x0 = a*rho
#             y0 = b*rho
#             x1 = int(x0 + 1000*(-b))
#             y1 = int(y0 + 1000*(a))
#             x2 = int(x0 - 1000*(-b))
#             y2 = int(y0 - 1000*(a))
#         
#             cv2.line(gray,(x1,y1),(x2,y2),(0,0,255),2)
#     return gray
#     
#     
#     sift = cv2.SIFT()
#     kp = sift.detect(gray,None)
#     return cv2.drawKeypoints(gray,kp)
#     
#     #blurred_img = cv2.GaussianBlur(img,(5,5),0)
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     lower_yellow = np.array([0,50,50])
#     upper_yellow = np.array([70,255,255])
#     mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
#     kernel = np.ones((9,9),np.uint8)
#     closed_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#     return closed_mask
#     return cv2.bitwise_and(img,img, mask= closed_mask)
    
def display_image_and_wait(image):
    cv2.imshow('dst',image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
def print_rgb_hist(img, mask):
    color = ('b','g','r')
    hist = []
    for i,col in enumerate(color):
        color_hist = cv2.calcHist([img],[i],mask,[4],[0,256])
        color_hist = [num[0] for num in color_hist]
        hist.append(color_hist)
    return hist
#        plt.plot(histr,color = col)
#        plt.xlim([0,256])
#    plt.show()
    
def crop_mask_waffle(img):
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
    return [cv2.bitwise_and(img,img, mask= largest_blob_mask), largest_blob_mask]

def main():
    waffle_folder_name = os.getcwd() + '/waffles_images'
    new_img_dir = os.getcwd() + '/modified_waffle_images'
    
    if not os.path.isdir(waffle_folder_name):
        os.makedirs(waffle_folder_name)
    if not os.path.isdir(new_img_dir):
        os.makedirs(new_img_dir)
        
    for index, pic in enumerate(os.listdir(waffle_folder_name)):
        loaded_pic = cv2.imread(waffle_folder_name + '/' + pic) 
        print 'reading: ' + waffle_folder_name + '/' + pic
        new_img_name = 'waffle_pic_'+ str(index) + '.jpg'
        cropped_pic = crop_mask_waffle(loaded_pic)[0]
        hist = print_rgb_hist(loaded_pic, crop_mask_waffle(loaded_pic)[1])
#        display_image_and_wait(cropped_pic)
        cv2.imwrite(new_img_dir + '/' + new_img_name,cropped_pic)
        
main()