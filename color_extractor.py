# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:44:59 2020

@author: malrawi
"""

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib
# matplotlib.use("TkAgg")
# backend string are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 
# 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
# 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import matplotlib.pyplot as plt


def RGB2HEX(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))

class ColorExtractor():
    ''' Extracts the colors of one person fashion items/clothing'''
    def __init__(self, image, masks, labels, masked_img, number_of_colors = 16, max_num_colors = 1):
        ''' Extracts the colors of one person fashion items/clothing'''
        # inputed
        self.labels = labels     
        self.number_of_colors = number_of_colors
        self.max_num_colors = max_num_colors
        # computed
        self.counts_top_k=[]
        self.colors_centers = []
        self.find_colors(image, masks, masked_img, number_of_colors, max_num_colors)
        
    def pie_cluster(self, image, figure_size=(9, 6), fname=None):
        for j in range(len(self.labels)):
            hex_colors = [RGB2HEX(self.colors_centers[j][i]) for i in self.counts_top_k[j].keys()]    
            plt.figure(figsize = figure_size)
            plt.suptitle( self.labels[j] , fontsize=16)
            plt.pie(self.counts_top_k[j].values(), labels = hex_colors, colors = hex_colors,
                    rotatelabels = False)
            plt.show()
            if fname:
                plt.savefig('./Figures/' + fname + '_ncol='+ str(self.number_of_colors) + '_' + self.labels[j] +'.png')
                image.save('./Figures/' + fname +'.jpg')

        
    def remove_image_background(self, image, mask):
        mask =  np.concatenate(mask)
        modified_image = image.reshape(image.shape[0]*image.shape[1], 3)  
        modified_image = modified_image[mask] # removing the background
        return modified_image
    
    def get_colors_cluster(self, image, number_of_colors):      
        clf = KMeans(n_clusters = self.number_of_colors, n_jobs=10,
                      max_iter=3000, n_init=16, init='k-means++') # we are using higher number of max_iter and n_init to ensure convergence
        pred_labels = clf.fit_predict(image)
        counts = Counter(pred_labels)    
        colors_centers = clf.cluster_centers_.round()        
        return counts, colors_centers
        
    def get_top_k_colors_sorted(self, counts, k=4):      
        cnt = counts.most_common() # counts is casted as a sorted dictionary here into cnt   
        ''' May be we select the top k as a list, we don't need the counter anymore' '''
          
        counts_top_k = Counter() # creating a new counter to store the top k colors
        for i in range(k):        
            key = cnt[i][0] # key , the negative to start from the end of the list, which has the highest value, list is sorted
            num_pixels = cnt[i][1] # value
            counts_top_k[key] = num_pixels
            
        return counts_top_k
            
    def find_colors(self, image, masks, masked_img, number_of_colors, max_num_colors):
        for j in range(len(self.labels)):
            image_no_bkg = self.remove_image_background(masked_img[j], mask = masks[j])
            counts, colors_centers = self.get_colors_cluster(image_no_bkg, number_of_colors=number_of_colors)  
            counts_top_k = self.get_top_k_colors_sorted(counts, k = max_num_colors)  
            self.counts_top_k.append(counts_top_k)
            self.colors_centers.append(colors_centers)
        
         
    def __getitem__(self, ind):
        return self.counts_top_k[ind],self.colors_centers[ind], \
                    self.labels[ind], self.masked_img[ind]
    

    def __len__(self): 
        return len(self.counts_top_k)

