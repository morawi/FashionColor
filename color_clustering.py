# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 17:12:45 2020

@author: malrawi

Color Clustering

https://mattdickenson.com/2018/11/18/gmm-python-pyro/
https://github.com/mcdickenson/em-gaussian/blob/master/em-gaussian-pyro.py
https://pyro.ai/examples/dirichlet_process_mixture.html
https://pyro.ai/examples/gmm.html
https://scikit-learn.org/stable/auto_examples/mixture/plot_concentration_prior.html#sphx-glr-auto-examples-mixture-plot-concentration-prior-py
https://optuna.org/?fbclid=IwAR2lOExJAyfp_5XZnyLh_HBtE8n4Mse4mBu4w4tF4R-SOs4DsxBInvfVXzY
"""

import argparse
import platform
from color_extractor import ColorExtractor as color_extractor_obj
# from clothcoparse_dataset import get_clothCoParse_class_names 
from prepare_data import get_ClothCoP_images_as_pack, get_images_from_wardrobe
from color_utils import get_dataset
from color_table import ColorTable
import gc
import time
from clothing_class_names import get_59_class_names

parser = argparse.ArgumentParser()

# parser.add_argument("--num_colors", type=int, default=32, help="number of epochs of training")
parser.add_argument("--max_num_colors", type=int, default=0, help="max number of colors the user wants")
parser.add_argument("--dataset_name", type=str, default="ClothCoParse", help="name of the dataset: {ClothCoParse, or Modanet}")
parser.add_argument("--HPC_run", default=False, type=lambda x: (str(x).lower() == 'true'), help="True/False; -default False; set to True if running on HPC")
parser.add_argument('--save_fig_as_images', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses a pretrained model")
parser.add_argument('--color_upr_bound', default=True, type=lambda x: (str(x).lower() == 'true'), help="True/False: default True; True uses color upper bound for each item")
parser.add_argument("--method", type=str, default="3D_1D", help="name of method: {3D_1D, or 3D}")
# parser.add_argument("--lr", type=float, default=0.005, help="learning rate")


cnf = parser.parse_args()

if platform.system()=='Windows':
    cnf.n_cpu= 0


''' # number of colors should be equal or more than three,  to count for background, 
if 2 is used, backgound will be mixed with the original color and will cause problems 
and incorrect results '''



def extract_colors(cnf):    
    dataset, class_names_and_colors = get_dataset(cnf)
    
    # ids  = range(len(dataset))
    # ids = [271]
    ids = [58]
    # ids =[192]
    # ids = [323]
    ids = [833] # dress
    # ids = [121] # white shirt, gray pants, 
    # ids =[0]
    # ids = [94]
    # ids = [907]
    # ids = [737]
    # ids = [505]    
    
    all_persons_items = []    
    for i in ids:    
        image, masked_img, labels, image_id, masks, img_name = dataset[i]   
        data_pack = get_ClothCoP_images_as_pack(masks, masked_img, labels, img_name)        
        one_person_clothing_colors = color_extractor_obj(cnf,  
                                                         class_names_and_colors,
                                                         data_pack)
        fname = data_pack['img_name'] if cnf.save_fig_as_images else None
        one_person_clothing_colors.pie_chart(image, fname=fname, figure_size=(4, 4))
        all_persons_items.append(one_person_clothing_colors)
    return all_persons_items, image


def fill_ClothCoP_color_table(cnf, out_file = 'ref_clothCoP'):    
    dataset, class_names_and_colors = get_dataset(cnf)    
    # ids = [271, 58, 371, 192]     
    ids = range(4, 1002) # all images
    
    color_table_obj = ColorTable(dataset.class_names, cnf.max_num_colors)    
    for i in ids: #  for i, data_item in enumerate(dataset):        
        print('processing person/catalogue', i)
        image, masked_img, labels, image_id, masks, img_name = dataset[i]   
        data_pack = get_ClothCoP_images_as_pack(masks, masked_img, labels, img_name)
        one_person_clothing_colors = color_extractor_obj(cnf,  
                                                         class_names_and_colors,
                                                         data_pack)                                                           
        color_table_obj.append(one_person_clothing_colors)
    color_table_obj.build_table(table_type='Catalogue')    
    color_table_obj.save(out_file + '.pkl')
    # obj.analyze() # draw color distributions
        
    return color_table_obj


def fill_and_build_wardrobe_color_table(cnf, user_name='malrawi'):  
    print('processing wardrobe of' , user_name)
    class_names_and_colors = get_59_class_names()      
    color_table_obj = ColorTable(list(class_names_and_colors), cnf.max_num_colors)  
    data_pack = get_images_from_wardrobe(user_name= user_name)
    for d_pack in data_pack:
        wardrobe_clothing_colors = color_extractor_obj(cnf,  
                                                     class_names_and_colors,
                                                     d_pack) 
        fname = d_pack['img_name'] if cnf.save_fig_as_images else None
        wardrobe_clothing_colors.pie_chart(None, fname=fname, figure_size=(4, 4))                                                          
        color_table_obj.append(wardrobe_clothing_colors)
    color_table_obj.build_table(table_type='Wardrobe')
    print('Now saving model as ...', user_name+'.pkl')
    color_table_obj.save(user_name+'.pkl')
        
    return color_table_obj
    
cnf.max_num_colors = 14
cnf.color_upr_bound = True # when True, extracted colors will be bounded by an upper bound, like, for skin; num of colors will be 1 and for dress will be high, like 17 
cnf.max_num_colors = cnf.num_colors if cnf.max_num_colors==0 else cnf.max_num_colors # we can reduce the number of clusters according to this upper value, regardless of the number of clusters 
cnf.find_no_clusters_method = 'None' # 'gmm' # {'kmeans', 'gmm', 'bgmm', 'None' } # gmm is the best for now, bgmm=Bayes GMM
cnf.clustering_method = 'kmeans'  # {'kmeans', 'fcmeans', 'gmm', 'bgmm'}
cnf.clsuter_1D_method='Diff'  # {'MeanSift', 'Diff', '2nd_fcm', 'None'}: for None, no 1D cluster will be applied
cnf.action = ['build color catalogue', 'build color from user wardrobe', 'test']
cnf.action = cnf.action[2]

print(cnf)


tic = time.time()
wardrobe_name = 'real-malrawi' #
if cnf.action == 'build color catalogue': 
    obj = fill_ClothCoP_color_table(cnf, out_file = 'ref_clothCoP.pkl')
elif cnf.action == 'test':
    extract_colors(cnf)    
else: 
    obj_user = fill_and_build_wardrobe_color_table(cnf, user_name=wardrobe_name)
    


# x, image = extract_colors(dataset, class_names_and_colors, cnf)
print('Elapsed time is:', time.time()-tic)
gc.collect()

# x[0][0]['skin']