# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:44:59 2020

@author: malrawi
"""

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
from color_utils import merge_clusters, RGB2HEX
 




class ColorExtractor():
    ''' Extracts the colors of one person fashion items/clothing'''
    def __init__(self, image, masks, labels, masked_img, cnf):
                                                       
        # inputed
        self.labels = labels     
        self.number_of_colors = cnf.num_colors
        self.max_num_colors = cnf.max_num_colors
        
        # computed
        self.top_k_colors=[]
        self.colors_centers = []                     
                
        if cnf.method == '3D_1D':
            self.find_colors_via_clustering_3D_1D_cntPix(masks, masked_img, 
                                                         cnf.number_of_colors, 
                                                         cnf.max_num_colors)
        elif cnf.method == '3D':              
            self.find_colors_via_clustering_3D_1D(masks, masked_img, cnf.number_of_colors, 
                                            cnf.max_num_colors, use_quantize=cnf.use_quantize)
       


    def pie_chart(self, image, figure_size=(9, 6), fname=None):
        
        for j in range(len(self.labels)):            
            hex_colors = [RGB2HEX(self.top_k_colors[j][key]) for key in self.top_k_colors[j].keys()]    
            plt.figure(figsize = figure_size)
            plt.suptitle( self.labels[j] , fontsize=16)
            num_pixel_percent =[k*100 for k in self.top_k_colors[j].keys()]            
            plt.pie(num_pixel_percent, labels = hex_colors, colors = hex_colors,
                    rotatelabels = False)            
                       
            if fname:
                plt.savefig('./Figures/' + fname + '_ncol='+ str(self.number_of_colors) + '_' + self.labels[j] +'.png')                
            plt.show()
            plt.close()
        if fname: image.save('./Figures/' + fname +'.jpg') # saving the original image for comparision
        plt.show()
        plt.close()
        
    
    
    def remove_image_background(self, image, mask):
        mask =  np.concatenate(mask)  # inpus mask is 2D array, output mask is a vector from concatnating the input 2D array
        image_no_bkg = image.reshape(image.shape[0]*image.shape[1], 3)  # reshape to a vector with triplet values as entry, each row has R, G, B values
        image_no_bkg = image_no_bkg[mask] # removing the background via the mask
        return image_no_bkg
    
        
    def get_colors_cluster(self, image, number_of_colors):      
        clf = KMeans(n_clusters = self.number_of_colors, n_jobs=10,
                      max_iter=3000, n_init=16, init='k-means++') # we are using higher number of max_iter and n_init to ensure convergence
        pred_labels = clf.fit_predict(image)
        counts = Counter(pred_labels)    
        colors_centers = np.uint8(clf.cluster_centers_.round())
        return counts, colors_centers
        

    def find_colors_via_clustering_3D(self, masks, masked_img, number_of_colors, max_num_colors, use_quantize=False):
        ''' Clustering 3D is based on three-values; R, G, B
        use_num_pixels_percentage (p) the percentage denotes the probability of a color, 
        since we are picking max_num_colors out of the available 
        number_of_colors the sum will not be 1, the sum will one
        if and only if max_num_colors equals number_of_colors 
        '''
        use_num_pixels_percentage = True 
        for label_id in range(len(self.labels)):   
            num_pixels_in_mask = np.sum(masks[label_id]) if use_num_pixels_percentage else 1
            image  = masked_img[label_id]
            if use_quantize:
                image = Image.fromarray(masked_img[label_id])
                image = image.quantize(colors=number_of_colors, method=None, kmeans=0, palette=None).convert('RGB')
                image = np.array(image)               
            
            image_no_bkg = self.remove_image_background(image, mask = masks[label_id])
            counts, colors_centers = self.get_colors_cluster(image_no_bkg, number_of_colors=number_of_colors)  
            top_k_colors = counts.most_common()[:max_num_colors] 
            x= dict(top_k_colors)            
            numpixels_and_colors = []       
            for key in x.keys(): numpixels_and_colors.append( [x[key]/num_pixels_in_mask, colors_centers[key]])                                   
            self.top_k_colors.append(dict(numpixels_and_colors))
    
    def find_colors_via_clustering_3D_1D(self, masks, masked_img, number_of_colors, max_num_colors):
        ''' Clustering_3D_1D is based on three-values; R, G, B, and then, on 1D via h value from hsv
        use_num_pixels_percentage (p) the percentage denotes the probability of a color, 
        since we are picking max_num_colors out of the available 
        number_of_colors the sum will not be 1, the sum will one
        if and only if max_num_colors equals number_of_colors 
        '''
        use_num_pixels_percentage = True 
        for label_id in range(len(self.labels)):   
            num_pixels_in_mask = np.sum(masks[label_id]) if use_num_pixels_percentage else 1
            image  = masked_img[label_id]           
            image_no_bkg = self.remove_image_background(image, mask = masks[label_id])
            
            # use clustering to reduce the number of colors
            counts_from_cluster, colors_centers = self.get_colors_cluster(image_no_bkg, number_of_colors=number_of_colors)  
            
            # find the average of close colors according to hue value
            merged_centers, merged_labels = merge_clusters(colors_centers, quantile= None) 
            counts_from_cluster = dict(counts_from_cluster.most_common())
                        
            numpixels =  np.zeros(len(counts_from_cluster), dtype = 'float'); updated_labels = []
            colors_= np.zeros(merged_centers.shape, dtype = 'uint8')            
            
            for i, key in enumerate(counts_from_cluster.keys()): 
                numpixels[i] =  counts_from_cluster[key]/num_pixels_in_mask 
                colors_ [i] =  merged_centers[key]
                updated_labels.append(merged_labels[key])
                
            pixsum = []; colors2 = []
            for i in np.unique(merged_labels):
                jj = updated_labels==i
                pixsum.append( sum(numpixels[jj]) )
                idxTrue = np.where(jj)[0][0]
                colors2.append( colors_[idxTrue])
                
            sorted_indxs = np.flip( np.argsort(pixsum) )
            if len(sorted_indxs)>max_num_colors: 
                sorted_indxs=sorted_indxs[:max_num_colors]                
            
            numpixels_and_colors = []   
            for i in sorted_indxs:
                numpixels_and_colors.append([pixsum[i], colors2[i]])
                            
            self.top_k_colors.append(dict(numpixels_and_colors))
    
    def find_colors_via_clustering_3D_1D_cntPix(self, masks, masked_img, number_of_colors, max_num_colors, use_quantize=False):
        ''' Clustering_3D_1D is based on three-values; R, G, B, and then, on 1D via h value from hsv
        use_num_pixels_percentage (p) the percentage denotes the probability of a color, 
        since we are picking max_num_colors out of the available 
        number_of_colors the sum will not be 1, the sum will one
        if and only if max_num_colors equals number_of_colors 
        '''
        use_num_pixels_percentage = True 
        for label_id in range(len(self.labels)):   
            num_pixels_in_mask = np.sum(masks[label_id]) if use_num_pixels_percentage else 1
            image  = masked_img[label_id]           
            image_no_bkg = self.remove_image_background(image, mask = masks[label_id])
            
            # use clustering to reduce the number of colors
            counts_from_cluster, colors_centers = self.get_colors_cluster(image_no_bkg, number_of_colors=number_of_colors)  
            
            # finding the percentage of the color, something like the probability
            for i, key in enumerate(counts_from_cluster.keys()): 
                counts_from_cluster[i] =  counts_from_cluster[i]/num_pixels_in_mask 
            
            # find the average of similar colors according to hue value
            grouped_centers, grouped_labels = merge_clusters(colors_centers, counts_from_cluster, quantile= None) 
            counts_from_cluster = dict(counts_from_cluster.most_common())
                        
            
            numpixels =  np.zeros(len(counts_from_cluster), dtype = 'float'); updated_labels = []
            colors_= np.zeros(grouped_centers.shape, dtype = 'uint8')      
            for i, key in enumerate(counts_from_cluster.keys()): 
                numpixels[i] =  counts_from_cluster[key] # /num_pixels_in_mask 
                colors_ [i] =  grouped_centers[key] # here we use grouped_centers instead of counts_from_cluster as we are interested in the new colors found according to the hue value
                updated_labels.append(grouped_labels[key]) # we will be neededing the labels later
            
            # grouping colors according to similar labels find via the hue value
            pixsum = []; colors_grouped = []
            for i in np.unique(grouped_labels):
                jj = updated_labels==i
                pixsum.append( sum(numpixels[jj]) )
                idxTrue = np.where(jj)[0][0]
                colors_grouped.append( colors_[idxTrue])
                
            # sorting the indices according to pixel count    
            sorted_indxs = np.flip(np.argsort(pixsum)) 
            if len(sorted_indxs)>max_num_colors: 
                sorted_indxs = sorted_indxs[:max_num_colors]                
            #  pixels sums and corresponding colors
            numpixels_and_colors = []   
            for i in sorted_indxs:
                numpixels_and_colors.append([pixsum[i], colors_grouped[i]])
                            
            self.top_k_colors.append(dict(numpixels_and_colors))

        
    def find_colors_via_quantize(self, masked_img, number_of_colors, max_num_colors, image):
        for j in range(len(self.labels)):            
            image = Image.fromarray(masked_img[j])
            x_img = image.quantize(colors=number_of_colors, method=None, kmeans=0, palette=None)
            z = x_img.convert('RGB').getcolors()
            z.sort()
            top_k_colors= z[-max_num_colors-1:-max_num_colors+1] 
            self.top_k_colors.append(dict(top_k_colors))
                
        
    def __getitem__(self, ind):
        item={}
        for j in range(len(self.labels)):               
            item[ self.labels[j] ] = self.top_k_colors[j]
        
        return item
    

    def __len__(self): 
        return len(self.max_num_colors)


