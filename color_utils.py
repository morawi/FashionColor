# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 17:45:22 2020

@author: malrawi

"""
import numpy as np
from clothcoparse_dataset import ImageDataset 
from sklearn.cluster import MeanShift, estimate_bandwidth

import cv2
# from modanet_dataset import ModanetDataset # this is becuase PyCocoTools is killing matplotlib backend
# https://github.com/cocodataset/cocoapi/issues/433


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

def RGB2HEX(rgb):
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def cluster_1D(x, quantile=None, show=False):   
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
    x = x.reshape(-1, 1)
    
    if quantile not in (0, None):
        bw = estimate_bandwidth(x, quantile=quantile, n_samples=len(x))
    else: bw = None
    
    ms = MeanShift(bandwidth=bw, bin_seeding=False) # bin_seeding = True causes a problem
    ms.fit(x)  
    result = {}
    result['labels'] = ms.labels_ 
    result['cluster_centers'] = ms.cluster_centers_                
    if show:
        print(result['cluster_centers'])
        print("number of estimated clusters : %d" % len(np.unique(result['labels'])))
        print(result['labels'])
        print('bw=', bw)    
    
    return result, ms


    
def merge_clusters(rgb_array_in, counts_from_cluster, quantile=None):
    rgb_array = rgb_array_in.copy() # we need to make a copy, and it has to be of float type to prevent overflow
    rgb_array = np.expand_dims(rgb_array, axis=0)
    hsv = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2HSV)
    hsv = np.squeeze(hsv, axis=0)
    h_val = hsv[:, 0] # getting the h value to be used in clustering
    result, ms = cluster_1D(h_val, quantile=quantile)    
    rgb_array = np.squeeze(rgb_array, axis=0)
    rgb_array = rgb_array.astype(float) 
    # rgb_array = average_similar_colors(rgb_array, result['labels'])   
    rgb_array = average_similar_colors2(rgb_array, counts_from_cluster, result['labels'])   
    
    return rgb_array, result['labels']
    
def average_similar_colors(rgb_array, labels):
    ''' Direct average between hue grouped colors '''
    for i in np.unique(labels):
        mean_rgb_at_i = np.mean(rgb_array[labels == i], axis=0)
        rgb_array[ labels==i ] = np.uint8( mean_rgb_at_i )               
    return rgb_array

def average_similar_colors2(rgb_array, counts_from_cluster, labels):
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
    
    
    