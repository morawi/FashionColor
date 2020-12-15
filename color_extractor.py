# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 16:44:59 2020

@author: malrawi

"""

import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
import matplotlib.pyplot as plt
from color_utils import merge_clusters, RGB2HEX # remove_image_background # , save_masked_image
from color_names import ColorNames
from fcmeans import FCM
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture


class ColorExtractor():
    ''' Extracts the colors of one person fashion items/clothing'''
    def __init__(self, cnf, 
                 class_names_and_colors,
                 data_pack):
        
        ''' Extracts the colors of one person fashion items/clothing'''
        # input        
        self.class_names_and_colors = class_names_and_colors
        self.color_upr_bound = cnf.color_upr_bound
        # self.number_of_colors = cnf.num_colors # I think this is not used any more
        self.max_num_colors = cnf.max_num_colors        
        self.color_names_obj = ColorNames()
        self.clustering_method = cnf.clustering_method
        self.clsuter_1D_method = cnf.clsuter_1D_method
        self.find_no_clusters_method = cnf.find_no_clusters_method     
        
        # computed values           
        self.item={}                          
        self.labels = data_pack['labels']        
        self.find_colors(data_pack)
         
        
    def pie_chart(self, image, figure_size=(9, 6), fname=None, save_image = False):        
        
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
                plt.savefig('./Figures/' + label_val+'-'+fname + '_ncol='+ '_' +'.png')            
            plt.close()
        if save_image: image.save('./Figures/' + fname +'.jpg') # saving the original image for comparision
        
        plt.close()
   
    
    
    def estimate_number_of_colors(self, image, num_clusters, 
                                  label_val, 
                                  clf_model, 
                                  plot_graph=True ):
        ''' Use GMM to estimate the optimum num of colors
        We can also use prior probability by making use of the item label. For example, a belt is more likely to
        have one color.
        Hence, we may use low number of colors for bags, shoe, skin, purse, hair, accessories.
        In addition, if the min estimated color value is far from the highest,
        we may select the min based on a threshold value 
        ''' 
               
        gm_scores = np.array([])                
        for no_clusters_idx in range(1,num_clusters+1):
            counts, colors_centers, clf = clf_model(image, no_clusters_idx, label_val)            
            if self.find_no_clusters_method == 'kmeans':
                gm_scores = np.append(gm_scores, abs(clf.score(colors_centers)/no_clusters_idx) )  # should be used wiht k-means
            else: 
                gm_scores = np.append(gm_scores,  abs(clf.score(colors_centers)))
                
            
        plot_graph=True  
        
        
        
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
        second_min = np.argmin(gm_scores[gm_scores != np.amin(gm_scores)])
        
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
            
            
        print('predicted no of colors', predicted_no_colors, '\n\n')              
        return predicted_no_colors        
        
    
    
    
    
        
    ''' 
    NOTE: It is important to always set the random state to a fixed value, 
    to enable reproducibility of results 
    '''
    def bgmm_model(self, image, no_clusters,label_val):
        ''' Runs Bayes GMM model and returns the colors and the counts (num of pixels) for each color'''
        
        print('Bayes GMM')
        bgmm = BayesianGaussianMixture(n_components=no_clusters, 
                                       random_state=1,
                                       n_init=1, # using more number of init is not good, will lead to overfitting
                                       covariance_type = 'tied', #  'tied', # 'full',
                                       #  mean_precision_prior=1e-2, 
                                       #  covariance_prior=1e0 * np.eye(3),
                                       #  weight_concentration_prior=0.005,                                      
                                       # max_iter=100
                                      )
        bgmm.fit(image)
        pred_labels  = bgmm.predict(image)
        counts = Counter(pred_labels)
        colors_centers = np.uint8(bgmm.means_.round()) #  gmm.means_ # 
        for i in range(0,len(colors_centers)): 
            if counts[i]==0: counts[i]=0 # if counts[i] does not exist (value 0, as the model will not detect the lable in the prediction if it is not in the data)
        return counts, colors_centers, bgmm    
    
    
    
    
    def gmm_model(self, image, no_clusters,label_val):
        ''' Runs GMM model and returns the colors and the counts (num of pixels) for each color'''
        
        gmm = GaussianMixture(n_components=no_clusters, 
                              random_state=1,
                              n_init=4, # using more number of init is not good, will lead to overfitting
                              covariance_type = 'tied', #{'full', 'tied', 'diag', 'spherical')
                              max_iter=100,
                              )
        gmm.fit(image)
        pred_labels  = gmm.predict(image)
        counts = Counter(pred_labels)
        colors_centers = np.uint8(gmm.means_.round()) #  gmm.means_ # 
        for i in range(0,len(colors_centers)): 
            if counts[i]==0: counts[i]=0 # if counts[i] does not exist (value 0, as the model will not detect the lable in the prediction if it is not in the data)
        return counts, colors_centers, gmm
   
    
   
    
    
    def kmm_model(self, image, no_clusters,label_val):
        ''' Runs k-means model and returns the colors and the counts (num of pixels) for each color '''
        clf = KMeans(n_clusters = no_clusters, 
                     n_jobs=30, 
                     n_init=24, # 8
                     random_state=1,
                     tol=1e-4, 
                     max_iter=100, 
                     init='k-means++')
                         # we are using higher number of max_iter and n_init for better convergence
        pred_labels = clf.fit_predict(image)
        counts = Counter(pred_labels)    
        colors_centers = np.uint8(clf.cluster_centers_.round())
        return counts, colors_centers, clf
        
    
    
    def get_k_model_name(self):
        ''' Rerunrs the model to be used, the input is the model name '''
        if  self.find_no_clusters_method == 'kmeans':
            clf = self.kmm_model 
        elif self.find_no_clusters_method == 'gmm':  
            clf = self.gmm_model
        elif self.find_no_clusters_method == 'bgmm':
            clf = self.bgmm_model
            
        return clf
 
        
    def get_colors_cluster(self, image, no_clusters, clustering_method, label_val):
        ''' Retunrs the color centers and the count for each color. It uses GMM, Kmeans and FCmeans
        the inputs are:
            - image (as a vector of RBG values, no background)
            - no_clusters: The number of colors (K value)
            - clustering_method
            - label_val: This is the clothing label name, e.g. t-shirt, dress, pants, etc.
        '''                
        num_clusters = self.class_names_and_colors[label_val] if self.color_upr_bound else no_clusters
        if self.find_no_clusters_method == 'None':            
            no_colors = num_clusters # else, use the input number of clusters            
        else: 
            no_colors = self.estimate_number_of_colors(image, num_clusters, 
                                                       label_val, self.get_k_model_name())            
            
        
        counts=dict([]); colors_centers=[] # forward assignmet                
        if clustering_method == 'kmeans':
            counts, colors_centers,_ = self.kmm_model(image, no_colors, label_val)        
        elif clustering_method == 'gmm':  
            counts, colors_centers,_ = self.gmm_model(image, no_colors, label_val)
        elif clustering_method == 'bgmm':  
            counts, colors_centers,_ = self.bgmm_model(image, no_colors, label_val)
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
         
    
    def sum_pixels_based_on_new_labels(self, colors_, grouped_labels, updated_labels, numpixels):
        ''' sum the pixels according to the new labeling in updated_labels found via the hue value '''
        pixsum = []; colors_grouped = []
        for i in np.unique(grouped_labels):
            jj = updated_labels==i
            pixsum.append( sum(numpixels[jj]) )
            idxTrue = np.where(jj)[0][0]
            colors_grouped.append(colors_[idxTrue])
            
        return pixsum, colors_grouped
       
       
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
        
        pixsum, colors_grouped = self.sum_pixels_based_on_new_labels(colors_, grouped_labels, updated_labels, numpixels)
             
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
        colr_cutoff_probability = 0.5; numpixels_and_colors = [];  total_sum = 0
        
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
    
        
    
            
        
    def find_colors(self, data_pack):        
        self.item['im_name'] = data_pack['img_name'] # adding the file name to be able to track the results if needed
        if len(data_pack['labels'])==1: # if there is only one item per image, a wardrobe case
            label_val = data_pack['labels'][0]
            self.item[label_val] = self.find_colors_via_clustering(data_pack['all_images_no_bkg'], 
                                            data_pack['num_pixels_in_items'], 
                                            label_val)
        else: # for multi items per image 
            for class_id, label_val in enumerate(data_pack['labels']):  
                self.item[label_val] = self.find_colors_via_clustering(data_pack['all_images_no_bkg'][class_id], 
                                        data_pack['num_pixels_in_items'][class_id], 
                                        label_val)
        
        
    
    def find_colors_via_clustering(self, image_no_bkg, num_pixels_in_items, label_val):
        '''  Clustering_3D_1D is based on three-values; R, G, B, and then, on 1D via h value from hsv
        use_num_pixels_percentage (p) the percentage denotes the probability of a color, 
        since we are picking max_num_colors out of the available 
        number_of_colors the sum will not be 1, the sum will one
        if and only if max_num_colors equals number_of_colors    '''       
        
        counts_from_cluster, colors_centers = self.get_colors_cluster(image_no_bkg, 
                                                                      no_clusters=len(image_no_bkg),
                                                                      clustering_method = self.clustering_method,
                                                                      label_val=label_val)            
        # normalizing the counts to values between 0 and 1
        for i, key in enumerate(counts_from_cluster.keys()): 
            counts_from_cluster[i] =  counts_from_cluster[i]/num_pixels_in_items
         
        pixsum, colors_grouped = self.get_colors_via_hue(colors_centers, counts_from_cluster)            
        numpixels_and_colors = self.remove_low_probability_colors(pixsum, colors_grouped) 
           
        return numpixels_and_colors
                 
                
    def __getitem__(self):   
        return self.item       

    def __len__(self): 
        return len(self.item)


'''        # Depreciated ... not useful ... might try it later      
    def find_colors_via_quantize(self, masked_img, number_of_colors, max_num_colors, image):
        for j in range(len(self.labels)):    
            image = Image.fromarray(masked_img[j])
            x_img = image.quantize(colors=number_of_colors, method=None, kmeans=0, palette=None)
            z = x_img.convert('RGB').getcolors()
            z.sort()
            top_k_colors= z[-max_num_colors-1:-max_num_colors+1] 
            top_k_colors.append(dict(top_k_colors))
    
'''

# import matplotlib
# matplotlib.use("TkAgg")
# backend string are ['GTK3Agg', 'GTK3Cairo', 'MacOSX', 'nbAgg', 'Qt4Agg', 
# 'Qt4Cairo', 'Qt5Agg', 'Qt5Cairo', 'TkAgg', 'TkCairo', 'WebAgg', 'WX', 'WXAgg',
# 'WXCairo', 'agg', 'cairo', 'pdf', 'pgf', 'ps', 'svg', 'template']