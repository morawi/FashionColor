# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:12:45 2020

@author: malrawi

Color Clustering

"""


from sklearn.cluster import KMeans

# import matplotlib
# matplotlib.use('TkAgg')
# import matplotlib.pyplot as plt
 
# import tkinter
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import cv2
from collections import Counter
from clothcoparse_dataset import ImageDataset 
from modanet_dataset import ModanetDataset 

import argparse
import platform
import datetime
import calendar
import os
import sys
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--num_epochs", type=int, default=300, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset: ClothCoParse or Modanet ")
parser.add_argument("--batch_size", type=int, default=2, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.005, help="learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--evaluate_interval", type=int, default=50, help="interval between sampling of images from generators")
parser.add_argument("--checkpoint_interval", type=int, default=50, help="interval between model checkpoints")
parser.add_argument("--train_percentage", type=float, default=0.9, help="percentage of samples used in training, the rest used for testing")
parser.add_argument("--experiment_name", type=str, default=None, help="name of the folder inside saved_models")
parser.add_argument("--print_freq", type=int, default=100, help="progress print out freq")
parser.add_argument("--lr_scheduler", type=str, default='OneCycleLR', help="lr scheduler name, one of: OneCycleLR, CyclicLR StepLR, ExponentialLR ")
parser.add_argument("--job_name", type=str, default='test', help=" name for the job used in slurm ")

parser.add_argument("--HPC_run", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default False; set to True if running on HPC")
parser.add_argument("--remove_background", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; - default False; set to True to remove background from image ")
parser.add_argument("--person_detection", default=False, type=lambda x: (str(x).lower() == 'true'), help=" True/False; - default is False;  if True will build a model to detect persons")
parser.add_argument("--train_shuffle", default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default True to shuffle training samples")
parser.add_argument("--redirect_std_to_file", default=False, type=lambda x: (str(x).lower() == 'true'),  help="True/False - default False; if True sets all console output to file")
parser.add_argument('--pretrained_model', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses a pretrained model")

opt = parser.parse_args()

if platform.system()=='Windows':
    opt.n_cpu= 0



def RGB2HEX(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

# not in use at the moment
def remove_repeated_colors(center_colors, counts):     
    unq, cnt = np.unique(center_colors, axis=0, return_counts=True)
    repeated_groups = unq[cnt > 1]

    for repeated_group in repeated_groups:
        repeated_idx = np.argwhere(np.all(center_colors == repeated_group, axis=1))
        print(repeated_idx.ravel())

def remove_background(image, mask):
    mask =  np.concatenate(mask)
    modified_image = image.reshape(image.shape[0]*image.shape[1], 3)  
    modified_image = modified_image[mask] # removing the background
    return modified_image
    
def get_colors_cluster(image, number_of_colors):      
    clf = KMeans(n_clusters = number_of_colors, n_jobs=10,
                  max_iter=3000, n_init=16, init='k-means++') # we are using higher number of max_iter and n_init to ensure convergence
    labels = clf.fit_predict(image)
    counts = Counter(labels)    
    center_colors = clf.cluster_centers_.round()          
    return counts, center_colors


def pie_cluster(counts, center_colors, figure_size=(9, 6), show_image=False, label=''):
    hex_colors = [RGB2HEX(center_colors[i]) for i in counts.keys()]
    if show_image: plt.imshow(image)
    plt.figure(figsize = figure_size)
    # plt.Text('my fig')
    plt.suptitle( label , fontsize=16)
    plt.pie(counts.values(), labels = hex_colors, colors = hex_colors)
    # plt.show()


def remove_background_color(counts, center_colors, background_thr=10):    
    sum_colors= np.sum(center_colors, axis=1)         
    if min(sum_colors)<background_thr:
         background_id =  np.argmin(sum_colors)
         center_colors = np.delete(center_colors, background_id, axis=0)
         counts.pop(background_id)   
    else: print('Could not remove background, you need to use hihger background threshold')
    return counts

def print_colors(counts, center_colors):
    for i in counts.items(): 
        key= i[0]; no_pixels = i[1]
        print('there are', no_pixels, 'pixels of', center_colors[key], RGB2HEX(center_colors[key]) )
        
        
def get_one_image():
    ''' This can be used for debugging only'''
    image_name = 'yello_jacket_box.bmp' # 'jaket_box.jpg'
    path2image = 'C:/MyPrograms/Data/testing_images/'
    image = cv2.imread(path2image+image_name); 
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image



def get_top_k_colors_sorted(counts, center_colors, k=4):    
    
    cnt = counts.most_common() # counts is casted as a sorted dictionary here into cnt             
    counts_top_k = Counter() # creating a new counter to store the top k colors
    for i in range(k):        
        key = cnt[i][0] # key , the negative to start from the end of the list, which has the highest value, list is sorted
        num_pixels = cnt[i][1] # value
        counts_top_k[key] = num_pixels
        
    return counts_top_k

if opt.dataset_name=='ClothCoParse':
    dataset = ImageDataset("../data/%s" % opt.dataset_name, 
                            transforms_ = None, 
                            transforms_target= None,
                            mode="train",                          
                            HPC_run=opt.HPC_run, 
                            remove_background = opt.remove_background,   
                            person_detection = opt.person_detection
                        )
else:
    dataset = ModanetDataset("../data/%s" % opt.dataset_name, 
                             transforms_ = None,                             
                             HPC_run= opt.HPC_run, )
        


''' # number of colors should be equal or more than three,  to count for background, 
if 2 is used, backgound will be mixed with the original color and will cause problems 
and incorrect results '''
number_of_colors = 16
max_num_colors = 2
save_fig_as_images = True
fig_nm = 'ts'

if max_num_colors>number_of_colors:
    print('max_num_colors should be less than or equal than number_of_colors')
    exit()

''' # sum of pixel values over the three channel of the lowest center, for example, 
the values 1, 3, 0 will be detected as background if less than bachground_thr '''
background_thr = 1 #30 


for i in range(1):
    i=124
    image, masked_img, labels, image_id, masks = dataset[i]
    # image2 = get_one_image()
    for j in range(len(labels)):
        image_no_bkg = remove_background(masked_img[j], mask = masks[j])
        counts, center_colors = get_colors_cluster(image_no_bkg, number_of_colors=number_of_colors)
        # counts = remove_background_color(counts, center_colors, background_thr=background_thr) # only if remove_background not used
        counts_top_k = get_top_k_colors_sorted(counts, center_colors, k= max_num_colors)
        pie_cluster(counts_top_k, center_colors, label=labels[j])
        print_colors(counts_top_k, center_colors)    
        
        print('label', labels[j], ' detected top-k colors', counts_top_k)
        if save_fig_as_images:
            plt.savefig('./Figures/'+fig_nm+'_'+labels[j] +'.png')
