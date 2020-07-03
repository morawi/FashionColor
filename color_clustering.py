# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:12:45 2020

@author: malrawi

Color Clustering

"""

import argparse
import platform
from color_extractor import ColorExtractor as color_extractor_obj
from color_utils import get_dataset
from color_table import ColorTable


parser = argparse.ArgumentParser()

parser.add_argument("--num_colors", type=int, default=32, help="number of epochs of training")
parser.add_argument("--max_num_colors", type=int, default=2, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset: ClothCoParse or Modanet ")
parser.add_argument("--HPC_run", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default False; set to True if running on HPC")
parser.add_argument('--save_fig_as_images', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses a pretrained model")
# parser.add_argument("--lr", type=float, default=0.005, help="learning rate")


cnf = parser.parse_args()

if platform.system()=='Windows':
    cnf.n_cpu= 0


''' # number of colors should be equal or more than three,  to count for background, 
if 2 is used, backgound will be mixed with the original color and will cause problems 
and incorrect results '''

if cnf.max_num_colors > cnf.num_colors:
    print('max_num_colors should be less than or equal than number_of_colors')
    exit()




def generate_all_colors(dataset, cnf):    
    all_persons_items = []    
    # ids  = range(len(dataset))
    # ids = [271]
    # ids = [58]
    ids =[192]
    ids = [323]
    
    for i in ids:    
        image, masked_img, labels, image_id, masks, im_name = dataset[i]   
        one_person_clothing_colors = color_extractor_obj(image, masks, labels, masked_img, 
                                                         number_of_colors = cnf.num_colors, 
                                                         max_num_colors = cnf.max_num_colors,
                                                         use_quantize=cnf.use_quantize,
                                                         method = cnf.method,
                                                         image_name=im_name)  
        
        fname = im_name if cnf.save_fig_as_images else None
        one_person_clothing_colors.pie_chart(image, fname=fname, figure_size=(4, 4))
        all_persons_items.append(one_person_clothing_colors)
    return all_persons_items, image


def fill_color_table(dataset, cnf):        
    ids = [271]
    ids = [271, 58]
    color_table_obj = ColorTable(dataset.class_names, cnf)    
    for i in ids: #i, data_item in enumerate(dataset)   
        image, masked_img, labels, image_id, masks, im_name = dataset[i]   
        one_person_clothing_colors = color_extractor_obj(image, masks, labels, masked_img, 
                                                         number_of_colors = cnf.num_colors, 
                                                         max_num_colors = cnf.max_num_colors,
                                                         use_quantize=cnf.use_quantize, 
                                                         image_name = im_name)  
        color_table_obj.add_fashion_item(one_person_clothing_colors)
    return color_table_obj
    

cnf.method = '3D_1D' # methods are: {'3D_1D', '3D'}
cnf.num_colors = 16 # we perhapse need to use different set of colors depending on the item, for skin should be 4
cnf.max_num_colors= 4
cnf.use_quantize = True
dataset = get_dataset(cnf)
obj = fill_color_table(dataset, cnf)
obj.build_table()
# x, image = generate_all_colors(dataset, cnf)

# cnf.method = '3D' # methods are: {'3D_1D', '3D'}
# x, image = generate_all_colors(dataset, cnf)
# x[0][0]['skin']


