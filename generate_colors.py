# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:53:57 2020

@author: malrawi
"""

# https://color.adobe.com/create/color-wheel
# very nice illustration of colors and their complements, analogous,  https://www.canva.com/colors/color-wheel/

# on color Complementary vs color opposite https://www.quora.com/What-is-the-complementary-colour-of-black
# analogous https://www.tigercolor.com/color-lab/color-theory/color-harmonies.htm
# color opposite, althoiug callled complementary https://www.101computing.net/complementary-colours-algorithm/
# color complement code   https://stackoverflow.com/questions/40233986/python-is-there-a-function-or-formula-to-find-the-complementary-colour-of-a-rgb
# there is a nice hue wheel that shows how much color changes with the angle
# split complementary https://colorswatches.info/split-complementary-colors
# https://www.pinterest.co.uk/pin/390546598913400705/

#Complete the code here...
from colorsys import rgb_to_hsv, hsv_to_rgb
import numpy as np
from color_utils import RGB2HEX
from color_names import ColorNames
import matplotlib.pyplot as plt
import itertools
import sys


def find_split_points(width_on_wheel=None, split_step = 2):  
    ''' width_on_wheel is angular width on color wheel '''
    xx = (9+split_step)/100 if width_on_wheel==None else width_on_wheel 
    xx = int(np.ceil(100*xx/split_step)*split_step) # correcting xx so that it accepts division by split_step
    zz= np.arange(-xx, 0, int(xx/split_step)) / 100    
    theta = list(zz)-np.flipud(zz) 
    return theta


def get_theta(n_step):    
    n_step = 3 if n_step <3 else n_step # will have problems if n_step less than 3, zero division
    split_start = 0
    split_end = 128 #  255
    stepee= (split_end-split_start)/n_step   
    theta = np.floor(np.arange(split_start, split_end, stepee))
    return theta


def generate_probability(num_thetas, use_orig_prob, keep_original_color, p0, p_major):
    num_thetas = num_thetas + int(keep_original_color)
    if use_orig_prob: 
        prob = [p0]*(num_thetas)
    else:
        if keep_original_color: 
            prob = [(1-p_major)/(num_thetas-1)]*(num_thetas-1) + [p_major]
        else:
            prob = [1/num_thetas]*num_thetas           
    return prob

def list_of_rgb_to_names(col_):
    color_names_obj = ColorNames()     
    hex_colors = [ RGB2HEX(col_val) for col_val in col_ ]
    names_of_colors = [color_names_obj.get_color_name(item) for item in  hex_colors]  
    return names_of_colors # , hex_colors


def color_pie_chart(design_pattern, figure_size=(9, 6), fname=None):          
        pixel_counts, hex_colors = zip(*[( design_pattern['prob'][i], RGB2HEX(design_pattern['color'][i])) for i in range(len(design_pattern['prob'])) ])      
        # color_names_obj = ColorNames()    
        # names_of_colors = [color_names_obj.get_color_name(item) for item in  hex_colors]            
        pixel_counts_labels =[ str(int(np.round(100*item, 2)))+'%' for item in pixel_counts]   # color names from color_names class can be added here
        # print(names_of_colors)
        plt.figure(figsize = figure_size)
        # plt.suptitle(label_val, fontsize=22)
        plt.pie(100*np.array(pixel_counts), labels = pixel_counts_labels, colors = hex_colors,
                rotatelabels = False, textprops={'fontsize': 18})            
        # plt.pie(num_pixel_percent, labels = hex_colors, colors = hex_colors, # drawing color vlaues in hexdecimal
        #         rotatelabels = False)       
        
        if fname:    
            plt.savefig('./Figures/' + fname + '.png')        
        plt.show()                               
        plt.close()

def get_alpha_beta_if_plane_color(theta, hsv, intensity_low_threshold, verbose,
                                  intensity_high_threshold=240, saturation_threshold=0.1):
    
    alpha=0; beta=0
    if theta>0: 
        if hsv[1]<saturation_threshold: 
            if verbose: print('Color purity (saturation) is low: increasing it to 0.2')
            # rgb  = color_monochromatic(rgb_in, np.ceil(255*abs(theta)))
            alpha= 0.2
        if hsv[2]<intensity_low_threshold:
            if verbose: print('value/intensity is low increading it to', intensity_low_threshold)
            beta =  intensity_low_threshold # lighten the color
        if hsv[2]>intensity_high_threshold:
            if verbose: print('value/intensity is high decreading it by', -40)
            beta = -40 # darken the color by using a -ve beta, to be subtracted from hsv[2] later
    return alpha, beta

    
def color_opposite(rgb):
  return [255-rgb[0], 255-rgb[1], 255-rgb[2]]


def apply_opposite(design_pattern_copy):
    design_pattern = design_pattern_copy.copy()
    col_=[]
    for rgb in design_pattern['color']:
        col_opposite = color_opposite(rgb)
        col_.append(col_opposite)        
    design_pattern['color'] = col_
    design_pattern['design_name']  = [design_pattern['design_name'][0] + '-opposite']
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    return design_pattern
    


def color_complementary(rgb_in, theta=0.5, verbose = True): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
    """returns RGB components of complementary color"""
    intensity_low_threshold = 40
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])
    alpha, beta = get_alpha_beta_if_plane_color(theta, hsv, intensity_low_threshold, verbose)                
    rgb = hsv_to_rgb((hsv[0] + theta) % 1, (hsv[1]+alpha)%1, (hsv[2] + beta)%256)
    return list(np.ceil(rgb)) # we should in fact use np.ceil for more accuracy
        

def color_shade(rgb_in, shade=10): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
        """returns RGB components of complementary color"""
        hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])        
        rgb = hsv_to_rgb(hsv[0], hsv[1], (hsv[2] + shade)%256 )
        return [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def apply_shade(design_pattern_copy, shade_val): # shade_val between 0 and 255
    design_pattern = design_pattern_copy.copy()
    col_=[]
    for rgb in design_pattern['color']:
        col_shade = color_shade(rgb, shade=shade_val)
        col_.append(col_shade)        
    design_pattern['color'] = col_
    design_pattern['design_name']  = [design_pattern['design_name'][0]  + '-shade']
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    return design_pattern
    

def generate_the_colors(rgb, theta, prob, design_name, 
                        color_function=color_complementary, keep_original_color=False):
    col_=[]; 
    design_pattern = {}    
    for th in theta:
        col_.append(color_function(rgb, th))
    design_pattern['color'] = col_ #  np.array(col_, dtype =float) 
    design_pattern['prob']  = prob # np.array(prob)
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    design_pattern['design_name'] = [design_name]
    if keep_original_color:
        design_pattern['color'].append(rgb)
        design_pattern['color_names'].append(list_of_rgb_to_names([rgb])[0])
    return design_pattern


def min_array_diff(zz):
    xx = np.abs([zz[0]-zz[1], zz[0]-zz[2], zz[1]-zz[2]])
    return np.min(xx)
    

def color_purity(rgb_in, theta= 128, verbose=False, purity_threshold = 10): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
    """ Returns RGB by changing color purity (saturation) of rgb_in
    theta values not in use now, as we are using randomly generated s and v values
    
    """    
    intensity_low_threshold = 40
    min_diff_rgb = min_array_diff(rgb_in)    
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])
    alpha, beta = get_alpha_beta_if_plane_color(theta, hsv, intensity_low_threshold, verbose)        
    
    if min_diff_rgb>purity_threshold:
        rgb = hsv_to_rgb(hsv[0], (hsv[1] + theta/255)%1, (hsv[2]+beta)%256)   # we are not using alpha for saturatoin as theta is changing, and no need to change the hue
    else: # else, this is gray and purity will entirely destroy the color to red or other values, we will use chromatic
        rgb = hsv_to_rgb(hsv[0], hsv[1], (hsv[2] + theta)%256) # this value will change the monochrom / grayscale of the color    
    
   
    return list(np.ceil(rgb).astype(int))


def get_n_split_purity(rgb, perc_upper=None, n_step=0,  
                              keep_original_color = False, p0=1, use_orig_prob=True):   
    p_major = 1/ (2*n_step+1)
    theta = get_theta(n_step)   
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major) 
               
    color_values = generate_the_colors(rgb, theta, prob, design_name='purity', 
                             color_function=color_purity, keep_original_color=keep_original_color)
    
    return color_values

def color_monochromatic(rgb_in, theta= 0.5): 
    """  Returns RGB monochromatic component of rgb_in, by changing the v/value/intensity of hsv
    """
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])    
    # rgb = hsv_to_rgb(hsv[0], hsv[1], theta) 
    rgb = hsv_to_rgb(hsv[0], hsv[1], (hsv[2] + theta)%256) # this value will change the monochrom / grayscale of the color    
    return list(np.ceil(rgb)) 



def get_n_split_monochromatic(rgb, perc_upper=None, n_step=0, 
                              keep_original_color = False, p0=1, use_orig_prob=True):    
    p_major = 1/ (2*n_step+1)
    theta = get_theta(n_step)    
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                
    return generate_the_colors(rgb, theta, prob, design_name='monochromatic', 
                               color_function=color_monochromatic, keep_original_color=keep_original_color)


def get_n_split_complementary(rgb, perc_upper=None, n_step=0, 
                              keep_original_color = False, p0=1, use_orig_prob=True):
    
    if n_step ==0: return generate_the_colors(rgb, theta=[0.5], prob=[p0], design_name= 'complement', 
                                              keep_original_color=keep_original_color) # this only returns the color complement         
    p_major = 1/ (2*n_step+1)
    theta = find_split_points(width_on_wheel=perc_upper, split_step = n_step) # width_on_wheel is angular width on color wheel
    theta = np.append(theta, 0) # this is necessary to get the complement of the exact color, as we are adding 0.5 below
    theta = [thx+0.5 for thx in theta] # converting to complement                   
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                
    return generate_the_colors(rgb, theta, prob, design_name='complement', keep_original_color=keep_original_color)
    

def get_n_split_analogous(rgb, perc_upper=None, n_step=0, 
                              keep_original_color = False, p0=1, use_orig_prob=True):    
    n_step = 1 if n_step ==0 else n_step 
    p_major = 1/ (2*n_step+1)    
    theta = find_split_points(width_on_wheel=perc_upper, split_step = n_step)    
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                
    return generate_the_colors(rgb, theta, prob, design_name='analogous', keep_original_color=keep_original_color)


    
        

def generate_pack_of_colors(rgb_val, n_split, p0=1, match_mode = 'complement', perc_upper=None):
    ''' 
    input - 
    match_mode
        - complement, complemen_opposite, complement_shade
        - analogous, analogous_opposite, analogous_shade, generate_pack_of_colors
    rgb_val: 
        a single RGB value example (255, 121, 11)
    output 
        - resutl: dictionary of each color set
    
    '''     
    # fname= str(rgb_val)+ '-'+ str(n_split)+'-split' 
    
    color_pack = {}
    if match_mode == 'purity':
        color_pack['purity'] = get_n_split_purity(rgb_val,                         
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0)
        return color_pack
    elif match_mode == 'monochromatic':
        color_pack['monochromatic'] = get_n_split_monochromatic(rgb_val, 
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0)
        return color_pack
        
    elif match_mode == 'opposite':
        color_pack['opposite'] = apply_opposite(generate_the_colors(rgb_val, theta=[0], prob=[p0], design_name= 'opposite'))
        return color_pack
    
    elif match_mode[:10]=='complement':
        design_pattern_comp= get_n_split_complementary(rgb_val, 
                            n_step=n_split, 
                            perc_upper = perc_upper,
                            p0=p0) # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)
        
        if match_mode == 'complement':
            color_pack['complement'] = design_pattern_comp       
        elif match_mode == 'complement_opposite':
            color_pack['complement_opposite'] = apply_opposite(design_pattern_comp) # print(n_split, 'alalogous-opposite') ; color_pie_chart(design_pattern_opp, fname='oppos_'+fname)       
        elif match_mode == 'complement_shade':  
            color_pack['complement_shade'] = apply_shade(design_pattern_comp, shade_val=128) #;print(n_split, 'opposite shade') ; # color_pie_chart(design_pattern_shade, fname='shade_'+fname)   
    
    elif match_mode[:9] == 'analogous':
        design_pattern_analog = get_n_split_analogous(rgb_val, 
                            n_step=n_split, 
                            perc_upper = perc_upper,
                            p0=p0) # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)
    
        
        if match_mode == 'analogous':                
            color_pack['analogous'] =  design_pattern_analog
        elif match_mode == 'analogous_opposite':         
            color_pack['analogous_opposite'] = apply_opposite(design_pattern_analog) # print(n_split, 'alalogous-opposite') ; color_pie_chart(design_pattern_opp, fname='oppos_'+fname)
        elif match_mode == 'analogous_shade':  
            color_pack['analogous_shade'] = apply_shade(design_pattern_analog, shade_val=128) #;print(n_split, 'opposite shade') ; # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
               
            
    return color_pack
        
        



# def color_complementary_old(rgb_in, theta=0.5, verbose = True): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
#     """returns RGB components of complementary color"""
#     intensity_low_threshold = 40
#     hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])
#     if hsv[1]<0.1 or hsv[2]<intensity_low_threshold: 
#         if verbose: print('Satuaration is 0 or intensity is low: complemntary not resolved, will change to monochromatic')
#         rgb  = color_monochromatic(rgb_in, np.ceil(255*abs(theta)))
#     else:
#         rgb = hsv_to_rgb((hsv[0] + theta) % 1, hsv[1], hsv[2])
#     return list(np.ceil(rgb)) # we should in fact use np.ceil for more accuracy
#     # return [int(rgb[0]), int(rgb[1]), int(rgb[2])] 
