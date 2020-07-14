# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:45:22 2020

@author: malrawi

"""
import numpy as np
from clothcoparse_dataset import ImageDataset 
from sklearn.cluster import MeanShift, estimate_bandwidth
import cv2
import matplotlib.pyplot as plt
from collections import Counter

# from modanet_dataset import ModanetDataset # this is becuase PyCocoTools is killing matplotlib backend
# https://github.com/cocodataset/cocoapi/issues/433

def RGB2HEX(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))



def get_dataset(opt):    
    if opt.dataset_name=='ClothCoParse':
        dataset = ImageDataset(root= "../data/%s" % opt.dataset_name,                                 
                                mode="train",                          
                                HPC_run=opt.HPC_run,                                 
                            )
    # else:
    #     dataset = ModanetDataset("../data/%s" % opt.dataset_name, 
    #                              transforms_ = None,                             
    #                              HPC_run= opt.HPC_run, )
    return dataset

def mean_rgb_colors(c1, c2):
    r1, g1, b1 = c1
    r2, g2, b2 = c2
    r = round((r1+r2)/2)
    g = round((g1+g2)/2)
    b = round((b1+b2)/2)
    
    return (r,g,b)



def cluster_1D(x, show=False):   
    ''' ---------------------------------------------------------------
    Cluster a 1D array into segments according to the quantile value, the 
    higher the quntile, the less number of clusters, and vice versa 
    
    input: 
        x - a 1D array to be clustered
        quantile - the divisions used to cluster x; the lower the quantile value, 
        the more number of clusters
        bandwidth will be automatically detected using quantile
        bw - bandwidth, not used for now as quantile will be used to estimate bw,
        the lower the bw value, the more clusters will result, 
        using any value for bw other than None will disable bw estimation
               
    output:
        - result['labels'] # a vector of all the labels having cluster id
        - result[cluster_centers] # a vector of cluster centers    
    -------------------------------------------------------------------- '''    
   
   
    ''' Best way is to estimate the quantile 
    from the variance of x 
    if x is low, then, quanitle should be high
    
    '''   
    quantile=0; max_iter = 0; bw = 0
    z = np.sort(x[:,0])
    v1 = np.std(z[z>np.mean(z)]) 
    v2 =  np.std(z[z<np.mean(z)])      
    
    if max(v1,v2)<3: quantile = 0.85 + min(v1,v2)/100
    if max(v1,v2)<6: quantile = 0.8
    elif max(v1,v2)<8: quantile = 0.75        
    if max(v1,v2)<10: quantile = 0.65        
    elif max(v1, v2) < 14: quantile = 0.5
    else: quantile = 0.05            
    while (bw == 0):    
        quantile = quantile + 0.05 #  0.1
        bw = estimate_bandwidth(x, quantile=quantile, n_samples=len(x)) 
        max_iter +=1
        if max_iter>100: 
            bw = None                
            break
    bw= 1
    ms = MeanShift(bandwidth=bw, bin_seeding=False) # bin_seeding = True causes a problem sometimes when bw is None (bw None is the default value)
    ms.fit(x)  
    result = {}
    result['labels'] = ms.labels_ 
    result['cluster_centers'] = ms.cluster_centers_                
    if show:
        print(result['cluster_centers'])
        print("number of estimated clusters : %d" % len(np.unique(result['labels'])))
        print(result['labels'])
        print('bw=', bw)    
    
    print('bw=',bw, 'quantile=', quantile, 'v1', v1, 'v2', v2, 'std', np.std(z))         
    print(ms.labels_)
    
    return result, ms

def differntial_1D_cluster(inp_vector, counts_from_cluster):    
    ''' Totally unspervised, all what is needed is one threshold
    value. No quantiles, no bandwidths, no iterations, nothing 
    '''
    # std =  np.std(inp_vector)  
    # mean = np.mean(inp_vector) + 0.001  # the 0.0001 to prevent zero division  if the mean is 0
    # coeff_var = std/mean
    # print('cf',coeff_var, 'std', std, 'mean', mean, np.median(inp_vector) , np.max(inp_vector))
    if np.max(inp_vector)-np.min(inp_vector)< 20:
        threshold = 30 
    else:        
        threshold = np.max(inp_vector)* len(inp_vector) / (np.sum(inp_vector)-len(inp_vector)*np.min(inp_vector))
    
    inp_ordered = np.sort(inp_vector)
    inp_appended = np.sort(np.append(inp_ordered, inp_ordered[0]))
    index_of_clusters = (inp_ordered  - inp_appended[:-1]) >  threshold        
    # Every True is followed by the members of the cluster, 
    # until the next True that marks the next cluster, and so on
    
    
    peak_idx = np.where(index_of_clusters == True)[0]
    labels = []; jj= 0; label_id=0
    for peak_id in peak_idx:
        labels.extend((peak_id-jj)*[label_id])
        label_id +=1
        jj=peak_id    
    # adding the last element manually    
    if len(labels)!= len(inp_vector):
        labels.extend( (len(inp_vector)-jj )*[label_id])
    
    # Now, reoredering labels to undo sorting
    I = np.argsort(inp_vector) # sorting indices
    J = np.argsort(I)  # these are the indices to undo sorting
    labels= np.array(labels)
    labels = labels[J] 
    
    # zz = Counter(labels)
    # most_occuring = zz[max(zz.values())]
    # other_indices  = labels[labels!=4]
    
    # used for debugging 
    # plt.plot(inp_ordered) 
    # plt.show()
    # print('new thresh', threshold)
    
            
    return labels


def merge_clusters(rgb_array_in, counts_from_cluster):
    rgb_array = rgb_array_in.copy() # we need to make a copy, and it has to be of float type to prevent overflow
    rgb_array = np.expand_dims(rgb_array, axis=0)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV) # COLOR_BGR2HLS
    # hsv = cv2.cvtColor(rgb_array, cv2.COLOR_BGR2HLS)
    hsv = np.squeeze(hsv, axis=0)            
    labels = differntial_1D_cluster(hsv[:, 0], counts_from_cluster) # hsv[:, 0] is the hue component
    rgb_array = np.squeeze(rgb_array, axis=0).astype(float)    
    rgb_array = average_similar_colors_pix_cnt(rgb_array, counts_from_cluster, labels)     
    return rgb_array, labels
    

def merge_clusters_old(rgb_array_in, counts_from_cluster):
    rgb_array = rgb_array_in.copy() # we need to make a copy, and it has to be of float type to prevent overflow
    rgb_array = np.expand_dims(rgb_array, axis=0)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hsv = np.squeeze(hsv, axis=0)    
    h_val=hsv[:, 0:1] # h_val = hsv[:, 0] # getting the h value to be used in clustering     
    result, ms = cluster_1D(h_val, counts_from_cluster)
    rgb_array = np.squeeze(rgb_array, axis=0).astype(float)    
    rgb_array = average_similar_colors_pix_cnt(rgb_array, counts_from_cluster, result['labels'])     
    return rgb_array, result['labels']
def average_similar_colors(rgb_array, labels):
    ''' Direct average between hue grouped colors '''
    for i in np.unique(labels):
        mean_rgb_at_i = np.mean(rgb_array[labels == i], axis=0)
        rgb_array[ labels==i ] = np.uint8( mean_rgb_at_i )
    return rgb_array

def average_similar_colors_pix_cnt(rgb_array, counts_from_cluster, labels):
    ''' Take the percentage average between grouped colors according to the
    count of pixels in each color in the group, example:
        0.75 has color (10, 15, 20) and 0.25 has color (20, 25, 10), the 
        weighted average will then be, 
        0.75*(10, 15, 20) + 0.25*(20,25,10) = [(30, 45, 60)+(20, 25, 10)]/4
        = (50, 70, 70)/4= (12.5, 17.5, 17.5).round()= (12, 17, 17)
        '''
    for i in np.unique(labels):
        idx = labels==i
        idd=np.where(idx==True)[0]        
        pixtotal=0; sum_rgb=0
        for jj in idd:                
            pix = counts_from_cluster[jj]
            sum_rgb += rgb_array[jj]*pix
            pixtotal += pix
        for jj in idd: # we have to fill all the jj elements to keep up with the original labels
            rgb_array[jj] = sum_rgb/pixtotal # non need to dvide by len(idd) as the sum of pix/pixtotal should be 1, total color area
        
    return np.uint8(rgb_array.round())

def ttest(x, idxs):
    ''' Equal or unequal sample sizes, similar variances 
    https://en.wikipedia.org/wiki/Student%27s_t-test
    '''
    out_t = 0
    num_idxs = len( np.unique(idxs) ) + 1 
    for i1 in range(num_idxs):
        for i2 in range(i1+1, num_idxs):
            x1= x[idxs==i1]
            x2= x[idxs==i2]
            m_diff = np.abs( np.mean(x1)-np.mean(x2) )
            sd1=np.std(x1); sd2=np.std(x2)
            n1=len(x1); n2=len(x2)
            N = np.sqrt(1/n1 + 1/n2)
            sp= np.sqrt( ((n1-1)*sd1*sd1 + (n2-1)*sd2*sd2) / (n1+n2-2) )
            t =  m_diff/(sp*N)
            print(t)
            out_t += t
    print(out_t)
    return out_t
    

''' This std/mean is not doing well, 
    all the problem was in estimating the quantile
    as shown above in cluster_1D() function'''
    # std =  np.std(rgb_array_in, axis=1)  
    # mean = np.mean(rgb_array_in, axis=1) + 0.001  # the 0.0001 to prevent zero division  if the mean is 0
    # coeff_var = 10*(std/mean)
    # print(rgb_array_in)
    # print(coeff_var)
    # h_val[:,2:3] =  coeff_var[:, None]