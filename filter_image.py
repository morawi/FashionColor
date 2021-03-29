# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 14:09:49 2021

@author: malrawi
"""


import cv2
import numpy as np
import PIL.Image
from io import BytesIO
import IPython.display




def showarray(a, fmt='png'):
    a = np.uint8(a)
    f = BytesIO()
    PIL.Image.fromarray(a).save(f, fmt)
    IPython.display.display(IPython.display.Image(data=f.getvalue()))
    
# https://www.programmersought.com/article/9812128243/
def bilateral_meanshift_filter(image, mode='RGB'): 
    # input: img in RGB mode as numpy uint8 array
    if mode != 'RGB': image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    img, box = extract_image(image) # sppedup and accuracy: extract the image if it is embbeded in a large zero mask, this will speedup the implementation if we have images embbeded into larger zero-value ones, which is the case in semantic segmentaion
    mask = img>0 # will be used to mask out edge distortions due to filtering
    
    # img = cv2.bilateralFilter(src=img, d=-1, sigmaColor=100, sigmaSpace=15)
    img = cv2.bilateralFilter(img,9,75,75)
    
    '''
         Gaussian bilateral blur, equivalent to microdermabrasion
         Src: original image
         d: the field diameter of the pixel, which can be calculated by sigmaColor and sigmaColor
         sigmaColor: the standard deviation of the color space, generally the bigger the better
         sigmaSpace: the standard deviation of the coordinate space (pixel units), generally the smaller the better
    '''
        
    # img = cv2.pyrMeanShiftFiltering(src=img, sp=15, sr=20 )# cv2.pyrMeanShiftFiltering(src=img, sp=15, sr=20)
    '''
         Mean offset filter processing, want to be the operation of turning pictures into oil painting
         Src: original image
         Sp: the radius of the space window (The spatial window radius)
         Sr: the radius of the color window (The color window radius)
         Edge-preserving filtering by mean migration sometimes leads to excessive image blurring
    '''
    img = img*mask # removing some unwanted values at the edge due to filtering
    
    image[box['ymin']:box['ymax'],box['xmin']:box['xmax']] = img # place back the image to where it should be
    
    return image
           

def extract_image(image):
    box={}
    pos = np.where(image)
    box['xmin'] = np.min(pos[1])
    box['xmax'] = np.max(pos[1])
    box['ymin'] = np.min(pos[0])
    box['ymax'] = np.max(pos[0])
    
    return image[box['ymin']:box['ymax'],box['xmin']:box['xmax']], box
    

        
        
        
        