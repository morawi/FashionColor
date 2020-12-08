# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 15:51:45 2020

@author: malrawi
"""
import numpy as np
from PIL import Image
from os import listdir
import cv2
 
from os import path 


def remove_image_background(image, mask):
    '''' Removes the (2D) image background according to the mask,
    and returns the  the image as an array of RGB triplets '''
    mask =  np.concatenate(mask)  # inpus mask is 2D array, output mask is a vector from concatnating the input 2D array
    image_no_bkg = image.reshape(image.shape[0]*image.shape[1], 3)  # reshape to a vector with triplet values as entry, each row has R, G, B values
    image_no_bkg = image_no_bkg[mask] # removing the background via the mask
    return image_no_bkg



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
        

def get_images_from_wardrobe(path_to_wardrobe= 'C:/MyPrograms/Data/Wardrobe/', user_name='malrawi'):
    ''' labels after cleaning, as we will be using them to reason fashion matching; like, 
    shirt goes with a pant, but shirt does not go with a shirt, and the likes '''
    path_to_folder = path_to_wardrobe + user_name
    fnames = listdir(path_to_folder) # each image named as a lable, skir.jpg; jacket.jpg, etc
    data_pack=[]         
    
    for fname in fnames:
        record = {}
        label_val = path.splitext(fname)[0]   # removing the extensoin to get the label
        
        img  = Image.open(path_to_folder+'/'+fname)
        mask = np.array(img.getchannel('A').convert('1'), dtype=np.bool_) # the alpha band is used to store the mask        
        img  = np.array(img.convert('RGB'), dtype='uint8')
        
        record['all_images_no_bkg'] = remove_image_background(img, mask = mask)
        record['num_pixels_in_items'] = np.sum(mask)
        record['labels'] = [label_val[:label_val.find('_')] ] # removing what comes after, in the wardrobe, we may have several jackets, so, they are named jacket_1.png, jacket_2.png 
        record['img_name']= fname
        data_pack.append(record)          
    
    return data_pack

    

# data_pack = get_images_from_wardrobe(user_name='malrawi')

