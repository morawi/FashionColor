# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:51:45 2020

@author: malrawi
"""

from PIL import Image
from os import listdir, path
from filter_image import bilateral_meanshift_filter as img_filter, showarray 
import matplotlib.pyplot as plt
# import scipy.signal as sc
import numpy as np
from colors_from_hist import colors_via_3d_hist
import cv2
import time

# from colorsys import rgb_to_hsv

# from scipy.signal import savgol_filter
def remove_image_background(image, mask, filter_image=False, verbose=True):
    '''' image: numpy format
    Removes the (2D) image background according to the mask,
    and returns the  the image as an array of RGB triplets '''
    
    # image filter destroys the results ... dunno why     
    image = img_filter(image) if filter_image else image # no good results, histogram is destroyed. This would be slow if the image is segmented from the whole outfit      
    if verbose: showarray(image)
    
    mask =  np.concatenate(mask)  # inpus mask is 2D array, output mask is a vector from concatnating the input 2D array
    image_no_bkg = image.reshape(image.shape[0]*image.shape[1], 3)  # reshape to a vector with triplet values as entry, each row has R, G, B values
    image_no_bkg = image_no_bkg[mask] # removing the background via the mask
    peaked_colors = colors_via_3d_hist(image_no_bkg)
        
    return image_no_bkg, peaked_colors



def get_ClothCoP_images_as_pack(masks, masked_img, labels, img_name):
    all_images_no_bkg = []
    num_pixels_in_items = []
    data_pack={}
    for class_id, label_val in enumerate(labels):            
        image  = masked_img[class_id]           
        all_images_no_bkg.append(remove_image_background(image, mask = masks[class_id]) )
        num_pixels_in_items.append(np.sum(masks[class_id]))        
        # save_masked_image(image, label_val, path= 'C:/MyPrograms/FashionColor/Experiments/', img_name = 'tmp.png')
    data_pack['all_images_no_bkg'] = all_images_no_bkg
    data_pack['num_pixels_in_items'] = num_pixels_in_items
    data_pack ['labels'] = labels
    data_pack ['img_name'] = img_name 
    return data_pack

        

def get_images_from_wardrobe(path_to_wardrobe= 'C:/MyPrograms/Data/Wardrobe/', 
                             user_name='malrawi',
                             resize_enabler=True):
    ''' labels after cleaning, as we will be using them to reason fashion matching; like, 
    shirt goes with a pant, but shirt does not go with a shirt, and the likes '''
    path_to_folder = path_to_wardrobe + user_name
    fnames = listdir(path_to_folder) # each image named as a lable, skir.jpg; jacket.jpg, etc
    data_pack=[]         
    
    max_allowed_size = 160000 
    ''' if the area of the image is lower than this, 
    it will be scaled down to max_allowed_size. This is important to make sure
    the filter bank work correctly. As a large image may stretch the histogram
    and more peeks will be detected. One way to remove this threshold is by
    making adding more filter banks. Currently, this implementation only supports
    two filter banks, one for a histogram with bins smaller than 500, and 
    another if the bins are greater than 500. 
    See filter_bank_bins=500 in smooth_filter_1D of colors_from_hist.
    This resizing will also speedup the implementation.
    If one wants to keep the image size as is, perhapse performing image smoothing 
    would help.   '''
        
    
    for fname in fnames:
        record = {}
        label_val = path.splitext(fname)[0]   # removing the extensoin to get the label
        img  = Image.open(path_to_folder+'/'+fname)
        mask = np.array(img.getchannel('A').convert('1'), dtype=np.bool_) # the alpha band is used to store the mask
        img  = np.array(img.convert('RGB'), dtype='uint8')    
        img_size = img.shape[0]*img.shape[1]
        if (img_size)>max_allowed_size and resize_enabler is True:   # scale the image down, keep aspect ratio, if it larger than max_allowed_size         
            img, mask = resize_image(img, mask, scale_factor = np.sqrt(max_allowed_size/img_size))
            
        record['all_images_no_bkg'] = remove_image_background(img, mask = mask)
        record['num_pixels_in_items'] = np.sum(mask)
        record['labels'] = [label_val[:label_val.find('_')] ] # removing what comes after, in the wardrobe, we may have several jackets, so, they are named jacket_1.png, jacket_2.png 
        record['img_name']= fname
        data_pack.append(record)  
    
    return data_pack



def resize_image(img, mask, scale_factor = 0.5):           
    
    interpol_method = cv2.INTER_LINEAR_EXACT # cv2.INTER_NEAREST
        
    dsize = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))    
    img = cv2.resize(img, dsize=dsize, interpolation=interpol_method)
    mask = cv2.resize(1*mask, dsize= dsize, interpolation= interpol_method)
    mask = np.array(mask, dtype=bool)
    
    return img, mask

tic = time.time()
data_pack = get_images_from_wardrobe(user_name='malrawi-pick')
print('exec time', time.time()-tic)
# data_pack = get_images_from_wardrobe(user_name='malrawi')


