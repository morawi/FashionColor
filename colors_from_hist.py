# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 11:16:46 2021

@author: malrawi
"""

import matplotlib.pyplot as plt
import scipy.signal as sc
import numpy as np
import torch

# import platform
# if platform.system()=='Windows':
#     from numpy import histogramdd as histogramdd        
# else: # Linux
#     from jax.numpy import histogramdd as histogramdd # jax cannot be installed on Windows

import torchist 


def colors_via_3d_hist(image_no_bkg, verbose = False):
    ''' predicts the number of colors from the image RGB histogram        
    '''
    
    num_points_to_search_tail_peaks = 16 # to check if the left or the right tails have higher values than the detected peaks and then adding them to the peak list
    hist_as_1D, x_bins = get_3D_histogram(image_no_bkg)        
    zz, idx_of_aa = smooth_filter_1D(hist_as_1D, verbose=verbose) # smooth the histogram, zz is compressed, hence we need to the original idxs of aa   
    max_peaks = find_the_peaks(zz) # find peaks
    max_peaks = add_tails_to_peak(zz, hist_as_1D, max_peaks, num_points_to_search_tail_peaks) # adding the tails if they are small
    max_peaks = idx_of_aa[max_peaks]
    
    print('num peaks without augmentation:', len(max_peaks), max_peaks)
    max_peaks = add_extra_peaks(max_peaks, x_bins)
    print('num peaks:', len(max_peaks), max_peaks)
    
    peaked_colors = np.take(x_bins, max_peaks, axis=0)   # these can either be used directly, or to seed kmeans or GMM
    # print('peaked colors', peaked_colors)
    
    return peaked_colors

    
def add_extra_peaks(max_peaks, x_bins):
    ''' randomly adding extra peaks to the discovered ones '''
    num_detected_colors = len(max_peaks)
    num_colors_to_add =1 +  np.ceil(num_detected_colors/4).astype(int) # adding 1 to the form, better be careful than be sorry
    peaks_to_add = np.random.randint(0, len(x_bins)-1, num_colors_to_add) 
    max_peaks = np.sort( np.concatenate((max_peaks, peaks_to_add)) )
    return max_peaks

    
def get_3D_histogram(image_no_bkg):
    # Build the histogram    
    H, _ = np.histogramdd(image_no_bkg, bins=255) # 255 is for 8-bit per pixel images, if it has higher pixels, this should be changed
    xx= np.where(H)        
    x_bins = np.hstack((xx[0].reshape(-1,1), xx[1].reshape(-1,1), xx[2].reshape(-1,1)))    
    hist_as_1D = H[xx]  # convert the histogram into a 1D vector  
    hist_as_1D = hist_as_1D/max(hist_as_1D) # then normalize it
    
    return hist_as_1D, x_bins

    
def convolve_it_numpy(vv, N, verbose=False):
    for n in N:
        if len(vv)<4*n: break
        vv = np.convolve(vv, np.ones(n)/n, mode='same')  
        if verbose: plt.figure(); plt.plot(vv)            
    return vv

def convolve_it_torch(vv, N):
    
    # making sure we have the right input
    if not torch.is_tensor(vv):
        vv = torch.tensor(vv)
    
    if not vv.is_cuda: # if it is not on cuda, place it on GPU 
        vv=vv.cuda()
        
    if len(vv.shape) == 1: # if it is not 3D, reshape it to 3D, that's needed for conv1d 
        vv= torch.reshape(vv, (1, 1, len(vv)))
                        
    # now, filtering 
    for n in N:
        if vv.shape[2]<4*n: break  
        kernel = kernel = torch.DoubleTensor([[ np.ones(n)/n ]]).cuda()  
        vv = torch.nn.functional.conv1d(vv, kernel, stride=1, padding = n//2) 

    return vv.squeeze().cpu().numpy()


    
def smooth_filter_1D(aa, reduce_peaks=True, use_torch=False, verbose = True):
    filter_bank_bins=[500]
    
    if verbose: plt.plot(aa); plt.show()
        
    if reduce_peaks: # if True, it only uses the peaks of the data, thus, reducint the curve profile
        idx_of_aa = sc.find_peaks(aa)[0]
        aa = aa[idx_of_aa]; print('only elite peaks')
    else:
        idx_of_aa = np.arange(0, len(aa))
    
    # choose torch or numpy
    filter_to_use = convolve_it_torch if use_torch else convolve_it_numpy
        
    # use a smaller filter bank if the signal is small sized    
    if len(aa)<filter_bank_bins[0]:
        N= [4, 8, 12, 16, 24, 32, 64, 56, 96 ]
    else: N= [64, 128, 96, 192 ]    
    
    zz = filter_to_use(aa.copy(), N)           
    zz= zz/max(zz) # normalize to 1    
    
    if verbose: plt.show(); plt.figure;plt.plot(aa); plt.plot(zz); plt.show()
    
    return zz, idx_of_aa
   
 

def find_the_peaks(zz, peak_priminence_threshold = 1, verbose=True):
    ''' Remove peaks with very low prominance, as a peak might be detected if
    it has a small width and amplitude comapared to its neighbors '''
    
    max_peaks, _ = sc.find_peaks(zz)    
    peak_prominences = sc.peak_prominences(zz, max_peaks) 
    peak_widths= sc.peak_widths(zz, max_peaks, prominence_data=peak_prominences, 
                                rel_height=1)[0]
    
    salient_peaks = peak_prominences[0]*peak_widths    
    max_peaks = max_peaks[salient_peaks>=peak_priminence_threshold]
    
    if verbose:
        print('----------')
        print("promin X widths", salient_peaks[salient_peaks>peak_priminence_threshold].astype(int) )
        
    return max_peaks





def add_tails_to_peak(zz, aa, max_peaks, num_points_to_search_tail_peaks):
    peak_ht_factor = 0.5 # if the tail is larger than the factor X all_peaks
    if max_peaks.size==0: # this can happen when there are no peaks, curve going down due to something, could be low number of pixels/points/colros
        # check if it is a vally
        min_peaks,_ = sc.find_peaks(1- zz)
        if min_peaks.size>0: # yep, this is a vally
            max_peaks = [0, len(zz)-1 ] # peaks are the two upper ridges of the vally
        else:       
            max_peaks = [np.argmax(aa)] # it is one peak (gaussian like, no tails) here we can probably use aa, and not zz
            
    else:        
        is_tail1_peak = np.argmax(zz[:num_points_to_search_tail_peaks]) # left tail
        is_tail2_peak = len(zz) - num_points_to_search_tail_peaks -1 + \
                np.argmax(zz[-num_points_to_search_tail_peaks:]) # right tail
        if (zz[is_tail1_peak]>(peak_ht_factor*zz[max_peaks])).any():
             if zz[is_tail1_peak]>zz[is_tail1_peak+1]: # it is a peak only if the curve is moving downwards starting from the detected tail
                max_peaks = np.concatenate(([is_tail1_peak], max_peaks))
                print('left tail added as it is a peak with amplitude', zz[is_tail1_peak])
        
        if (zz[is_tail2_peak]>zz[max_peaks]).any():  
            if  zz[is_tail2_peak]>zz[is_tail2_peak-1]: # it is a peak only if the curve is moving downwards starting from the detected tail
                max_peaks = np.append( max_peaks, is_tail2_peak)
                print('right tail added as it is a peak with amplitude', zz[is_tail2_peak])
                
    return max_peaks


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




