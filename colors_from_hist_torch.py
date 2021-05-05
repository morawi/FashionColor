# -*- coding: utf-8 -*-
"""
Created on Thu Apr 22 11:58:57 2021

@author: malrawi
"""
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:16:46 2021

@author: malrawi
"""

import matplotlib.pyplot as plt
import scipy.signal as sc
import numpy as np
import torch
import torchist
import kornia

# import platform
# if platform.system()=='Windows':
#     from numpy import histogramdd as histogramdd        
# else: # Linux
#     from jax.numpy import histogramdd as histogramdd # jax cannot be installed on Windows

def colors_via_3d_hist(image_no_bkg, verbose = False, use_feature_to_sort=True):
    ''' predicts the number of colors from the image RGB histogram        
    '''    
    num_points_to_search_tail_peaks = 16 # to check if the left or the right tails have higher values than the detected peaks and then adding them to the peak list
    hist_as_1D, x_bins = get_3D_histogram(image_no_bkg) 
    
    if use_feature_to_sort:
        hist_as_1D, x_bins, hue_sort_idx = sort_by_feature(hist_as_1D, x_bins, component_id=0) 
    impulse_prob, max_peaks = my_flatness(hist_as_1D)
    
    print('flatness', impulse_prob.cpu().numpy(), 'num peaks', max_peaks)
    if verbose: plt.plot(hist_as_1D.cpu().numpy()); plt.show()
        
    # max_peaks = has_distinctive_spikes(hist_as_1D)
       
    
    if max_peaks is None:
        main_peak = torch.argmax(hist_as_1D)               
        hist_as_1D, idx_hist_small = reduce_vector(hist_as_1D, reduce_peaks=True)    
        if verbose: plt.plot(hist_as_1D.cpu().numpy()); plt.show()
        hist_as_1D = smooth_filter_1D(hist_as_1D) # smooth the histogram, zz is compressed, hence we need to the original idxs of aa   
        if verbose: plt.plot(hist_as_1D.cpu().numpy()); plt.show()
        # max_peaks = find_the_peaks_torch(hist_small, width = 31).view(-1).cpu().numpy()        
        # max_peaks = add_tails_to_peak(hist_small.cpu().numpy(), main_peak, max_peaks, num_points_to_search_tail_peaks) # adding the tails if they are small
        # max_peaks = sc.argrelmax(hist_small.cpu().numpy(), axis=0, order=1, mode='wrap')[0] # find peaks, another
        
        hist_as_1D = hist_as_1D.cpu().numpy() # if the rest is numpy based
        max_peaks = find_the_peaks(hist_as_1D) # find peaks            
        max_peaks = add_tails_to_peak(hist_as_1D, main_peak, max_peaks, num_points_to_search_tail_peaks) # adding the tails if they are small
        max_peaks = idx_hist_small[max_peaks]
    
    print('num peaks:', len(max_peaks), max_peaks) if verbose else None
    # max_peaks = add_extra_peaks(max_peaks, num_bins = x_bins.shape[1]) 
    # print('num peaks with augment:', len(max_peaks), max_peaks) if verbose else None
    # if use_feature_to_sort: max_peaks = hue_sort_idx[max_peaks]
    peaked_colors = x_bins[:, max_peaks].T
    print(peaked_colors.cpu().numpy())
    return peaked_colors




def get_3D_histogram(image_no_bkg):
    ''' Build the histogram '''
    
    H = torchist.histogramdd(image_no_bkg.float().cuda(), bins=256, low=0., upp= 255.)  # low and upp: The lower and upper range of the bins. If not provided, range is simply (a.min(), a.max()) a is the input. Values outside the range are ignored. The first element of the range must be less than or equal to the second. range affects the automatic bin computation as well. While bin width is computed to be optimal based on the actual data within range, the bin count will fill the entire range including portions containing no data.  https://numpy.org/doc/stable/reference/generated/numpy.histogram.html
    x_bins = torch.where(H)
    hist_as_1D = H[x_bins]  # convert the histogram into a 1D vector     
    x_bins = torch.stack(list(x_bins), dim=0)   #   tuple_of_tensors_to_tensor()
    hist_as_1D = hist_as_1D/hist_as_1D.max() # then normalize it
    return hist_as_1D, x_bins 


def my_flatness(x):
    x = (6*x).exp()
    x=x/x.max()    
    impulse_prob = x[ x> ( x.mean() + x.std() ) ].mean()            
    the_peaks = (x>0.5).nonzero().view(-1).cpu().numpy() if impulse_prob>0.5 else None    
    return impulse_prob, the_peaks

    

# problem with this is that it returns duplicates RGBs
def sort_by_feature(hist, x_bins, transform=kornia.rgb_to_hsv, component_id=0):    
    xx = transform(x_bins.unsqueeze(dim=0).unsqueeze(dim=2)/255) # rgb data is assuemed to be in the range of (0, 1), hence we divide by 255 https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
    xx= xx.squeeze(dim=0).squeeze(dim=1)[component_id]
    _, sort_idx = torch.sort(xx)    
    hist = hist[sort_idx]
    x_bins = x_bins[:, sort_idx]    
    return hist, x_bins, sort_idx
    
        

 
def add_extra_peaks(max_peaks, num_bins):
    ''' randomly adding extra peaks to the discovered ones '''
    num_detected_colors = len(max_peaks)
    num_colors_to_add = 1 +  np.ceil(num_detected_colors/4).astype(int) # adding 1 to the form, better be careful than be sorry
    peaks_to_add = np.random.randint(0, num_bins - 1, num_colors_to_add) 
    max_peaks = np.sort( np.concatenate((max_peaks, peaks_to_add)) )
    return max_peaks
    

def convolve_it_torch(aa, N, num_pads = 100):    
        
    if len(aa.shape) == 1: # if it is not 3D, reshape it to 3D, that's needed for conv1d 
        aa= torch.reshape(aa, (1, 1, len(aa)))
    
    aa = torch.nn.functional.pad(aa, (num_pads, num_pads), value=0.0001)
    for n in N: # to speedup, filter bank can be generated at the start of the program and stored into a dictionary with n's as keys
        if aa.shape[2]<4*n: break  
        kernel = torch.FloatTensor([[ np.ones(n)/n ]]).cuda()  # maybe we can generate these filters in advance
        aa = torch.nn.functional.conv1d(aa, kernel, stride=1, padding = n//2) 
        
    aa=aa.squeeze()
    aa= aa[num_pads:len(aa)-num_pads]
    aa= aa/aa.max() # normalize to 1 # maybe it is better not to normalize to 1
    
    return aa


def reduce_vector(aa, reduce_peaks = True):         
    if reduce_peaks and len(aa)>300: # if True, it only uses the peaks of the data
        idx_of_aa = sc.find_peaks(aa.cpu().numpy())[0]
        # idx_of_aa = find_the_peaks_torch(aa, width = 3).view(-1).cpu().numpy()        
        # idx_of_aa = sc.argrelmax(aa.cpu().numpy(), axis=0, order=1, mode='wrap')[0] # find peaks, another
        aa = aa[idx_of_aa]        
    else:
        idx_of_aa = np.arange(0, len(aa))
    return aa, idx_of_aa
        


def smooth_filter_1D(aa, filter_bank_bins=[200, 500]): # bins should be sorted
            
    # use a smaller filter bank if the signal is small sized    
    ''' N should have odd values, to ensure the filter output
    has the same size of the input signal 
    this will ensure extracting the correct RGB value
    from the indices of the histogram ... 
    using even values can lead to changing the filter output
    and shifting thus the RGB values
    '''
    if len(aa)<filter_bank_bins[0]:
         N= [3, 7]
    elif len(aa)<filter_bank_bins[1]:
        N= [3, 7, 11, 15, 23, 31, 63, 55, 97 ]
    else: 
        N= [63, 127, 95, 191 ]  
        
    aa = convolve_it_torch(aa, N)           
        
    return aa
   
 

def find_the_peaks(aa, peak_priminence_threshold = 1, verbose=True):
    ''' Remove peaks with very low prominance, as a peak might be detected if
    it has a small width and amplitude comapared to its neighbors '''
    
    max_peaks, _ = sc.find_peaks(aa)    
    peak_prominences = sc.peak_prominences(aa, max_peaks) 
    peak_widths= sc.peak_widths(aa, max_peaks, prominence_data=peak_prominences, 
                                rel_height=1)[0]
    
    salient_peaks = peak_prominences[0]*peak_widths    
    max_peaks = max_peaks[salient_peaks>=peak_priminence_threshold]
    
    if verbose:        
        print("promin X widths", salient_peaks[salient_peaks>peak_priminence_threshold].astype(int) )
        
    return max_peaks

# https://discuss.pytorch.org/t/pytorch-argrelmax-or-c-function/36404/2
def find_the_peaks_torch(a, width = 31): # width = 31 # odd, can be changed if needed
    window_maxima = torch.nn.functional.max_pool1d_with_indices(a.view(1,1,-1), width, 1, padding=width//2)[1].squeeze()
    candidates = window_maxima.unique()
    nice_peaks = candidates[(window_maxima[candidates]==candidates).nonzero()]
    return nice_peaks



def add_tails_to_peak(zz, main_peak, max_peaks, num_points_to_search_tail_peaks):
    peak_ht_factor = 0.5 # if the tail is larger than the factor X all_peaks
    if max_peaks.size==0: # this can happen when there are no peaks, curve going down due to something, could be low number of pixels/points/colros
        # check if it is a vally
        min_peaks,_ = sc.find_peaks(1- zz)
        if min_peaks.size>0: # yep, this is a vally
            max_peaks = [0, len(zz)-1 ] # peaks are the two upper ridges of the vally
        else:       
            max_peaks = main_peak # the main peak it is one peak (could be gaussian like, no tails) 
            
    else:        
        is_tail1_peak = np.argmax(zz[:num_points_to_search_tail_peaks]) # left tail
        is_tail2_peak = len(zz) - num_points_to_search_tail_peaks -1 + \
                np.argmax(zz[-num_points_to_search_tail_peaks:]) # right tail
        if is_tail1_peak not in max_peaks and (zz[is_tail1_peak]>(peak_ht_factor*zz[max_peaks])).any():
             if zz[is_tail1_peak]>zz[is_tail1_peak+1]: # it is a peak only if the curve is moving downwards starting from the detected tail
                max_peaks = np.concatenate(([is_tail1_peak], max_peaks))
                print('left tail added as it is a peak with amplitude', zz[is_tail1_peak])
        
        if  is_tail1_peak not in max_peaks and (zz[is_tail2_peak]>peak_ht_factor*zz[max_peaks]).any():  
            if  zz[is_tail2_peak]>zz[is_tail2_peak-1]: # it is a peak only if the curve is moving downwards starting from the detected tail
                max_peaks = np.append( max_peaks, is_tail2_peak)
                print('right tail added as it is a peak with amplitude', zz[is_tail2_peak])
                
    return max_peaks

# # giving bad results https://dsp.stackexchange.com/questions/74772/how-to-check-if-a-signal-is-mainly-composed-of-a-few-impulse-peaks
# def find_flattness(x):
#     N= len(4*x)
#     x = x+1
#     flatness = N* (x.prod()).pow(1/N)/x.sum()
#     return flatness.cpu().numpy()
    
# def kurtosis(x):
#     # aa = torch.pow(x-x.mean(), 4.0)                   
#     # bb = torch.pow(x.std(), 4.0)                 
#     # val = (aa/bb).mean() - 3.0 
#     val = torch.pow((x-x.mean())/x.std(), 4).mean() #  - 3
    
#     return val

# def dd(xx):
#     return torch.stack(( xx[0].view(-1), xx[1].view(-1), xx[2].view(-1)), -1)    

# def tuple_of_tensors_to_tensor(tuple_of_tensors):
#     return  torch.stack(list(tuple_of_tensors), dim=0)    

# def has_distinctive_spikes(hist, thresh=0.05, num_spikes=30):    
#     max_spikes = (hist>thresh).nonzero().view(-1).cpu().numpy()
#     if len(max_spikes)>num_spikes:
#         max_spikes = None                    
#     return max_spikes


# # convert the histogram into a 1D vector, then normalize it
    # for i,j,k in zip(xx[0], xx[1], xx[2]): 
    #     aa = np.append(aa, H[i][j][k])


# mmx = np.argmax(aa)       
    # print(xx[0][mmx], xx[1][mmx], xx[2][mmx])
    # print(aa[mmx])
    

# def convolve_it(vv, N, verbose=True):
#     for n in N:
#         if len(vv)<4*N: break
#         vv = np.convolve(vv, np.ones(n)/n, mode='same')   
         
    
#     if len(vv)>4*N:   
#         vv = np.convolve(vv, np.ones(N)/N, mode='same')   
#     if verbose: plt.figure(); plt.plot(vv)            
#     return vv


# def rgb2hsv(rgb_val):
#         ''' Converst an array of a RGB values to HSV values
#         Input:
#             rgb_val: array of RGB values rgb_val=[[r1,g1, b1], [r2, g2, b2],...]
#         output: List of HSV values 
#         '''                
#         hsv_out = []
#         for rgb in rgb_val:            
#             hh = list(rgb_to_hsv(rgb[0],rgb[1], rgb[2]))                                    
#             hh[0]= int( 255*hh[0])
#             hh[1]= int( 255*hh[1])
            
#             hsv_out.append(hh)                           
            
#         return hsv_out



# def to_odd(num):
#     return int(num + num%2 - 1) # makding sure window size is odd



# smooth the histogram using a filter bank
    # if len(aa)<30000:
    #     window_size = to_odd(1500)  # we need to find the working range of window sizes, later      
    #     if len(aa)<500: # if aa length is small, let's use a low window size
    #         window_size = to_odd(int(len(aa)//1.5))
    #     elif len(aa)<window_size:
    #         window_size = to_odd(int(len(aa)//2))
    # zz = savgol_filter(aa, window_size, 1) # I'm using a filter bank, to acheive better smoothing
    # zz = savgol_filter(zz, to_odd(window_size//2), 1) # I'm using a filter bank, to acheive better smoothing
    # zz = savgol_filter(zz, to_odd(window_size//4), 2) # I'm using a filter bank, to acheive better smoothing
    
    # zz = savgol_filter(aa, window_size, filt_order-1) # I'm using a filter bank, to acheive better smoothing
    # zz = savgol_filter(zz, to_odd(window_size//1.5), filt_order)
    # zz = savgol_filter(zz, to_odd(window_size//2), filt_order)


# p1= (0,0)
# p2 = (0.33, no_points)
# slope = sly2-y1/ x2-x1 = (0.33-0)/(no_points-0) = 0.33/no_points 


# no_points = 100 
    # slope = aa[0]/no_points 
    # values_to_insert = slope * np.arange(0, no_points)
    # # aa =np.concatenate(([0]*no_points, aa))
    # aa = np.concatenate((values_to_insert, aa))




