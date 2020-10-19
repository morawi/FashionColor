# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:44:59 2020

@author: malrawi

"""

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
# import matplotlib
# matplotlib.use("TkAgg")
# backend string are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 
# 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
# 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']
import matplotlib.pyplot as plt
from PIL import Image
from color_utils import merge_clusters, RGB2HEX, remove_image_background # , save_masked_image
from color_names import ColorNames
from fcmeans import FCM
from sklearn.mixture import GaussianMixture
# from clothcoparse_dataset import get_clothCoParse_class_names # this function should be changed if using another dataset


class ColorExtractor():
    ''' Extracts the colors of one person fashion items/clothing'''
    def __init__(self, masks, labels, masked_img, cnf, class_names_and_colors,
                 image_name='', clustering_method='kmeans'):
        ''' Extracts the colors of one person fashion items/clothing'''
        # input        
        self.class_names_and_colors = class_names_and_colors
        self.color_upr_bound = cnf.color_upr_bound
        self.number_of_colors = cnf.num_colors
        self.max_num_colors = cnf.max_num_colors
        self.labels = labels
        self.color_names_obj = ColorNames()
        self.clustering_method = cnf.clustering_method
        self.clsuter_1D_method = cnf.clsuter_1D_method
        self.find_k_clustering_method = cnf.find_k_clustering_method
        
        
        # computed values           
        self.item={}                  
        self.item['im_name'] = image_name # adding the file name to be able to track the results if needed
        self.find_colors_via_clustering_3D_1D_cntPix(masks, masked_img)
         
        
    def pie_chart(self, image, figure_size=(9, 6), fname=None):        
        
        for j, label_val in enumerate(self.labels):                 
            pixel_counts, hex_colors = zip(*[( item[0], RGB2HEX(item[1])) 
                                             for item in self.item[label_val]])      
            
            names_of_colors = [ self.color_names_obj.get_color_name(item) for item in  hex_colors]            
            pixel_counts_labels =[ str(int(np.round(100*item, 2)))+'%' for item in pixel_counts]   # color names from color_names class can be added here
            print(label_val,': ', names_of_colors)
            plt.figure(figsize = figure_size)
            # plt.suptitle(label_val, fontsize=22)
            plt.pie(100*np.array(pixel_counts), labels = pixel_counts_labels, colors = hex_colors,
                    rotatelabels = False, textprops={'fontsize': 18})            
            # plt.pie(num_pixel_percent, labels = hex_colors, colors = hex_colors, # drawing color vlaues in hexdecimal
            #         rotatelabels = False)            
                                   
            if fname:    
                plt.savefig('./Figures/' + label_val+'-'+fname + '_ncol='+ str(self.number_of_colors) + '_' + self.labels[j] +'.png')            
            plt.close()
        if fname: image.save('./Figures/' + fname +'.jpg') # saving the original image for comparision
        
        plt.close()
   
    
    
    def estimate_number_of_colors(self, image, no_clusters, clustering_method, label_val, clf_model, 
                                  plot_graph=True ):
        ''' Use GMM to estimate the optimum num of colors
        We can also use prior probability by making use of the item label. For example, a belt is more likely to
        have one color.
        Hence, we may use low number of colors for bags, shoe, skin, purse, hair, accessories.
        In addition, if the min estimated color value is far from the highest,
        we may select the min based on a threshold value 
        ''' 
               
        gm_scores = np.array([])
        
        num_clusters = self.class_names_and_colors[label_val] if self.color_upr_bound else no_clusters
        for no_clusters_idx in range(1,num_clusters+1):
            counts, colors_centers, clf = clf_model(image, no_clusters_idx, label_val)                        
            gm_scores = np.append(gm_scores, abs(clf.score(colors_centers)) ) 
            for i in range(0,len(counts)): 
                if counts[i]==0: counts[i]=0 # if counts[i] does not exist (value 0 means it is empty), fill it with zero to prevent problems later                                   
        if plot_graph:
            plt.plot(gm_scores)           
            plt.rcParams.update({'font.size': 16})       
            plt.xticks(np.arange(0, num_clusters,2), np.arange(1, num_clusters+1, 2))            
            plt.xlabel('\n K')
            plt.ylabel('Score: Average log-likelihood \n') 
            plt.title( label_val+' : #pixels ' + str(len(image)))
            plt.show()               
        
        '''TODO- Algorithm to get K (no of colors) based on label and gmm_scores
        
           we need to add estimating the number of colros based on label-value
           logic, a purse is more probable to have 1 color
           skin can be 2 color, then discard the min probability, this is 
           due to several color outlier, lipstick, mostache, etc.
           etc. '''
        
        first_min = np.argmin(gm_scores)
        second_min = np.argmin(gm_scores[gm_scores !=np.amin(gm_scores)])
        
        if second_min < first_min:
            predicted_no_colors = second_min
        else:
            predicted_no_colors = first_min            
            
        predicted_no_colors = predicted_no_colors + 1 # since the array index starts in 0
        
        threshold_img_col = 100  # we need to better estimate this value, there seems to be a minimum number of samples to estimate the GMM        
        if len(image)/predicted_no_colors < threshold_img_col:
            print(len(image)/predicted_no_colors, 'below threshold', threshold_img_col)
            predicted_no_colors = 1 
            # when the number of pixels is low, estimation of number of colors might not be good
            
            
            
        return predicted_no_colors        
        
        
    def gmm_model(self, image, no_clusters,label_val):
        ''' Runs GMM model and returns the colors and the counts (num of pixels) for each color'''
        # https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html
        # # https://github.com/ocontreras309/ML_Notebooks/blob/master/GMM_Implementation.ipynb
        # covariance_type{'full' (default), 'tied', 'diag', 'spherical'}
        # In PyTorch (TODO): https://github.com/ldeecke/gmm-torch
        # https://pytorch.org/docs/stable/distributions.html
        # https://pyro.ai/examples/gmm.html
        # This  is the best: 
        #    https://pypi.org/project/pycave/
        gmm = GaussianMixture(n_components=no_clusters, random_state=1,
                                  n_init=2, # using more number of init is not good, will lead to overfitting
                                  covariance_type = 'tied',
                                  max_iter=100)
        gmm.fit(image)
        pred_labels  = gmm.predict(image)
        counts = Counter(pred_labels)
        colors_centers = np.uint8(gmm.means_.round()) #  gmm.means_ # 
        for i in range(0,len(counts)): 
            if counts[i]==0: counts[i]=0 # if counts[i] does not exist (value 0, not sure whay this is happening), fill it with zero to prevent problems later            
        return counts, colors_centers, gmm
    
    

        
    
    def kmm_model(self, image, no_clusters,label_val):
        ''' Runs k-means model and returns the colors and the counts (num of pixels) for each color '''
        clf = KMeans(n_clusters = no_clusters, n_jobs=10, n_init=15, 
                         random_state=1,
                         tol=1e-4, max_iter=300, init='k-means++')
                         # we are using higher number of max_iter and n_init for better convergence
        pred_labels = clf.fit_predict(image)
        counts = Counter(pred_labels)    
        colors_centers = np.uint8(clf.cluster_centers_.round())
        return counts, colors_centers, clf
        
        
    def get_colors_cluster(self, image, no_clusters, clustering_method, label_val): 
                
        
        if self.find_k_clustering_method != 'None':   
            clf = self.kmm_model if  self.find_k_clustering_method == 'kmeans'  else self.gmm_model
            no_colors = self.estimate_number_of_colors(image, no_clusters, clustering_method, label_val, 
                                                                 clf)
            print('predicted no of colors', no_colors, '\n\n')  
        else: 
            no_colors = no_clusters            
        
        counts=dict([]); colors_centers=[] # forward assignmet                
        if clustering_method == 'kmeans':
            counts, colors_centers,_ = self.kmm_model(image, no_colors, label_val)            
        
        elif clustering_method == 'gmm':  
            counts, colors_centers,_ = self.gmm_model(image, no_colors, label_val)      
            
        elif clustering_method == 'fcmeans':
            clf = FCM(n_clusters = no_colors)
            clf.fit(image)                      
            pred_labels  = clf.predict(image)
            counts = Counter(pred_labels)    
            colors_centers = np.uint8(clf.centers.round())        
                    
        else: 
            print('Error choosing clustering method')
            exit()
        
        return counts, colors_centers      
                
       
    def get_colors_via_hue(self, colors_centers, counts_from_cluster):
        ''' Find the average of similar colors according to hue value
            returns the grouped (final) colors, and the pixel sum in each color
        '''        
        if len(counts_from_cluster)> 1:         
            grouped_centers, grouped_labels = merge_clusters(colors_centers, counts_from_cluster, self.clsuter_1D_method)
        else:
            grouped_centers, grouped_labels = merge_clusters(colors_centers, counts_from_cluster) # None cluster_1D_method will be used
            
        counts_from_cluster = dict(counts_from_cluster.most_common()) # should this be put here or before????
        numpixels =  np.zeros(len(counts_from_cluster), dtype = 'float'); updated_labels = []
        colors_= np.zeros(grouped_centers.shape, dtype = 'uint8')      
        for i, key in enumerate(counts_from_cluster.keys()): 
            numpixels[i] =  counts_from_cluster[key]  
            colors_ [i] =  grouped_centers[key] # here we use grouped_centers instead of counts_from_cluster as we are interested in the new colors found according to the hue value
            updated_labels.append(grouped_labels[key]) # we will be neededing the labels later
        
        # grouping colors according to similar labels found via the hue value
        pixsum = []; colors_grouped = []
        for i in np.unique(grouped_labels):
            jj = updated_labels==i
            pixsum.append( sum(numpixels[jj]) )
            idxTrue = np.where(jj)[0][0]
            colors_grouped.append(colors_[idxTrue])
            
        return pixsum, colors_grouped
    
            
    def remove_low_probability_colors(self, pixsum, colors_grouped):
        ''' Remove colors that are trivial as their probability value
        is small. 
        If the color num of pixels is too small, that color needs to be removed: 
        For 20 colors, they will, in a typical situation have, 0.05 x 20 colors
        uniformly distributed. Setting the threshold at 0.5 means we can 
        remove at least 10 colors as (1/20)*10 = 0.5 if their share
        is somewhere less than 0.05 and keep the other 10;
        thus, retaining 10 other colors out of 20 '''
        
        #  pixels sums and corresponding colors
        colr_cutoff_probability = 0.5 # if self.clustering_method == 'gmm' else 0.5
        numpixels_and_colors = []
        total_sum = 0
        
        # sorting the indices according to pixel count    
        sorted_indxs = np.flip(np.argsort(pixsum)) 
        if len(sorted_indxs) > self.max_num_colors: 
            sorted_indxs = sorted_indxs[:self.max_num_colors]                
        
        # find the sum of all pixels after exclusing those with low probability
        for i in sorted_indxs:
            if pixsum[i]*self.max_num_colors > colr_cutoff_probability:                     
                total_sum += pixsum[i]
        
        # find the updated probability
        for i in sorted_indxs:
            if pixsum[i]*self.max_num_colors > colr_cutoff_probability:                 
                numpixels_and_colors.append([pixsum[i]/total_sum, colors_grouped[i]])   
                
        return numpixels_and_colors
    
    
    def find_colors_via_clustering_3D_1D_cntPix(self, masks, masked_img):
        '''  Clustering_3D_1D is based on three-values; R, G, B, and then, on 1D via h value from hsv
        use_num_pixels_percentage (p) the percentage denotes the probability of a color, 
        since we are picking max_num_colors out of the available 
        number_of_colors the sum will not be 1, the sum will one
        if and only if max_num_colors equals number_of_colors   '''
        
        for class_id, label_val in enumerate(self.labels):              
            
            image  = masked_img[class_id]           
            image_no_bkg = remove_image_background(image, mask = masks[class_id])
            # save_masked_image(image, label_val, path= 'C:/MyPrograms/FashionColor/Experiments/', img_name = 'tmp.png')
            
            
            
            # use clustering to reduce the number of colors
            if len(image_no_bkg) < self.number_of_colors:
                no_clusters = len(image_no_bkg)
            else: no_clusters = self.number_of_colors
            counts_from_cluster, colors_centers = self.get_colors_cluster(image_no_bkg, 
                                                                          no_clusters=no_clusters,
                                                                          clustering_method = self.clustering_method,
                                                                          label_val=label_val)            
            
            # finding the percentage of each color; the probability of each color
            use_num_pixels_percentage = True 
            num_pixels_in_mask = np.sum(masks[class_id]) if use_num_pixels_percentage else 1
            for i, key in enumerate(counts_from_cluster.keys()): 
                counts_from_cluster[i] =  counts_from_cluster[i]/num_pixels_in_mask 

             
            pixsum, colors_grouped = self.get_colors_via_hue(colors_centers, counts_from_cluster)
            

            
            numpixels_and_colors = self.remove_low_probability_colors(pixsum, colors_grouped)            
            self.item[label_val] = numpixels_and_colors
        
            
    def find_colors_via_quantize(self, masked_img, number_of_colors, max_num_colors, image):
        for j in range(len(self.labels)):    
            image = Image.fromarray(masked_img[j])
            x_img = image.quantize(colors=number_of_colors, method=None, kmeans=0, palette=None)
            z = x_img.convert('RGB').getcolors()
            z.sort()
            top_k_colors= z[-max_num_colors-1:-max_num_colors+1] 
            top_k_colors.append(dict(top_k_colors))
            
        
    def __getitem__(self):   
        return self.item       

    def __len__(self): 
        return len(self.item)


# This is not be needed, as the normalization will be don in get_colors_via_hue even if 
# the option is None (no hue grouping)
   

