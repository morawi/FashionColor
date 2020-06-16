# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:12:45 2020

@author: malrawi

Color Clustering

"""

from clothcoparse_dataset import ImageDataset 
from modanet_dataset import ModanetDataset 
import argparse
import platform
from color_extractor import ColorExtractor as color_extractor_obj


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


''' # number of colors should be equal or more than three,  to count for background, 
if 2 is used, backgound will be mixed with the original color and will cause problems 
and incorrect results '''
num_colors = 64
max_num_colors = 2
save_fig_as_images = True

if max_num_colors > num_colors:
    print('max_num_colors should be less than or equal than number_of_colors')
    exit()


def get_dataset(opt):    
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
    return dataset

def generate_all_colors(dataset, number_of_colors, max_num_colors):    
    all_persons_items = []    
    ids  =range(2) # range(len(dataset))
    ids = [975]
    for i in ids:    
        image, masked_img, labels, image_id, masks, im_name = dataset[i]   
        one_person_clothing_colors = color_extractor_obj(image, masks, labels, masked_img, 
                                                         number_of_colors = num_colors, 
                                                         max_num_colors = max_num_colors)  
        
        fname = im_name if save_fig_as_images else None
        one_person_clothing_colors.pie_cluster(image, fname=fname, figure_size=(4, 4))
        all_persons_items.append(one_person_clothing_colors)
    return all_persons_items, image
    
dataset = get_dataset(opt)
x, image = generate_all_colors(dataset, num_colors, max_num_colors)
