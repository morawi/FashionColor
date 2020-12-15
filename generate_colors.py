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

def generate_probability(num_thetas, use_orig_prob, keep_original_color, p0, p_major):
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
    
    

def color_complementary(rgb_in, theta=0.5, verose = True): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
    """returns RGB components of complementary color"""
    intensity_low_threshold = 40
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])
    if hsv[1]==0 or hsv[2]<intensity_low_threshold: 
        print('Satuaration is 0 or intensity is low: complemntary not resolved, will change to monochromatic')
        rgb  = color_monochromatic(rgb_in, np.ceil(255*theta))
    else:
        rgb = hsv_to_rgb((hsv[0] + theta) % 1, hsv[1], hsv[2])
    return [int(rgb[0]), int(rgb[1]), int(rgb[2])] 
    # return [list(np.ceil(rgb))] # we should in fact use np.ceil for more accuracy

        
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
    

def generate_the_colors(rgb, theta, prob, design_name, color_function=color_complementary):
    col_=[]; 
    design_pattern = {}    
    for th in theta:
        col_.append(color_function(rgb, th))
    design_pattern['color'] = col_ #  np.array(col_, dtype =float) 
    design_pattern['prob']  = prob # np.array(prob)
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    design_pattern['design_name'] = [design_name]
    return design_pattern


def color_monochromatic(rgb_in, theta= 0.5): 
    """ returns RGB components of monochromatic of rgb_in color         
    """
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])    
    rgb = hsv_to_rgb(hsv[0], hsv[1], (hsv[2]+theta)%256) # this value will change the monochrom / grayscale of the color
                        
    return [int(rgb[0]), int(rgb[1]), int(rgb[2])]


def color_purity(rgb_in, theta= 0.5): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
    """ returns RGB components of monochromatic of rgb_in color    
    theta values not in use now, as we are using randomly generated s and v values
    
    """    
    intensity_low_threshold = 30
    hsv = rgb_to_hsv(rgb_in[0], rgb_in[1], rgb_in[2])
    
    if hsv[1]==0 or hsv[2]<intensity_low_threshold: 
        print('Satuaration is 0 or intensity is low, purity not well resolved: will change to monochromatic')
        rgb  = color_monochromatic(rgb_in, np.ceil(255*theta))
    else:    
        rgb = hsv_to_rgb(hsv[0], (hsv[1]+theta)%1, hsv[2]) # this value will change the monochrom / grayscale of the color
                            
    return [int(rgb[0]), int(rgb[1]), int(rgb[2])]

def get_n_split_purity(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
                              keep_original_color = False, p0=1, use_orig_prob=True):    
    n_step = 2 if n_step <2 else n_step 
    split_start = 0; split_end = 255
    stepee= (split_end-split_start)/n_step
    startee = 0 if keep_original_color else stepee 
    theta = np.arange(startee, split_end, stepee)       
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                 
    return generate_the_colors(rgb, theta/255, prob, design_name='purity', color_function=color_purity)

def get_n_split_monochromatic(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
                              keep_original_color = False, p0=1, use_orig_prob=True):    
    n_step = 2 if n_step <2 else n_step 
    split_start = 0; split_end = 255
    stepee= (split_end-split_start)/n_step
    startee = 0 if keep_original_color else stepee 
    theta = np.arange(startee, split_end, stepee)       
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                
    
    return generate_the_colors(rgb, theta, prob, design_name='monochromatic', color_function=color_monochromatic)


def get_n_split_complementary(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
                              keep_original_color = False, p0=1, use_orig_prob=True):
    
    if n_step ==0: return generate_the_colors(rgb, theta=[0.5], prob=[p0], design_name= 'complement') # this only returns the color complement         
    theta = find_split_points(xx=perc_upper, split_step = n_step)                            
    theta = [thx+0.5 for thx in theta] # converting to complement               
    if keep_original_color: theta.append(0)         
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)                        
    return generate_the_colors(rgb, theta, prob, design_name='complement')
    

def get_n_split_analogous(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
                              keep_original_color = False, p0=1, use_orig_prob=True):    
    n_step = 1 if n_step ==0 else n_step 
    theta = find_split_points(xx=perc_upper, split_step = n_step)              
    if keep_original_color: theta.remove(0); theta.append(0) # moving zero to the end of list
    else: theta.remove(0) 
    prob = generate_probability(len(theta), use_orig_prob, keep_original_color, p0, p_major)            
    return generate_the_colors(rgb, theta, prob, design_name='analogous')


    
def find_split_points(xx=None, split_step = 2):  
    xx = (9+split_step)/100 if xx==None else xx #; xx = int(xx*100)    
    xx = int(np.ceil(100*xx/split_step)*split_step) # correcting xx so that it accepts division by split_step
    zz= np.arange(-xx, 0, int(xx/split_step)) / 100
    z2= -np.flipud(zz) 
    theta = list(zz)+list([0])+list(z2)     
    return theta
        

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
                        p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0)
        return color_pack
    elif match_mode == 'monochromatic':
        color_pack['monochromatic'] = get_n_split_monochromatic(rgb_val, 
                        p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0)
        return color_pack
        
    elif match_mode == 'opposite':
        color_pack['opposite'] = apply_opposite(generate_the_colors(rgb_val, theta=[0], prob=[p0], design_name= 'opposite'))
        return color_pack
    
    design_pattern_comp= get_n_split_complementary(rgb_val, 
                        p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0) # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)
           
    design_pattern_analog = get_n_split_analogous(rgb_val, 
                        p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                        n_step=n_split, 
                        perc_upper = perc_upper,
                        p0=p0) # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)

    
    if match_mode == 'complement':
        color_pack['complement'] = design_pattern_comp       
    elif match_mode == 'complement_opposite':
        color_pack['complement_opposite'] = apply_opposite(design_pattern_comp) # print(n_split, 'alalogous-opposite') ; color_pie_chart(design_pattern_opp, fname='oppos_'+fname)       
    elif match_mode == 'complement_shade':  
        color_pack['complement_shade'] = apply_shade(design_pattern_comp, shade_val=128) #;print(n_split, 'opposite shade') ; # color_pie_chart(design_pattern_shade, fname='shade_'+fname)   
    elif match_mode == 'analogous':                
        color_pack['analogous'] =  design_pattern_analog
    elif match_mode == 'analogous_opposite':         
        color_pack['analogous_opposite'] = apply_opposite(design_pattern_analog) # print(n_split, 'alalogous-opposite') ; color_pie_chart(design_pattern_opp, fname='oppos_'+fname)
    elif match_mode == 'analogous_shade':  
        color_pack['analogous_shade'] = apply_shade(design_pattern_analog, shade_val=128) #;print(n_split, 'opposite shade') ; # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
                
    # color_pack['num_split'] = [n_split]
        
    return color_pack
        
        




# def color_complement(rgb):
#     a, b, c=rgb[0], rgb[1], rgb[2]
#     if c < b: b, c = c, b
#     if b < a: a, b = b, a
#     if c < b: b, c = c, b
#     k=a+c
#     return [k - u for u in rgb]    
# print(color_complement(design_pattern['color'][0]  ))




# def get_n_split_complementary_old(rgb, perc_upper, n_step, p_major = 0.4, analogous=False):
    
#     design_name='analogous' if analogous else 'complement'
    
#     if n_step ==0: # this only returns the color complement 
#         return generate_the_colors(rgb, theta=[0, 0.5], prob=[0.5, 0.5], design_name= design_name)
#     theta = find_split_values(xx=perc_upper, split_step = n_step)
#     if not analogous:
#         theta.remove(0)
#         num_thetas = len(theta) # has to be without the complement angle        
#         theta =[0.5]+theta # adding the complement angle            
#         p_minor = (1-p_major)/num_thetas
#         prob=[p_major]+ [p_minor]*num_thetas
#     else:
#         num_thetas = len(theta) # has to be without the complement angle        
#         p_minor = (1-p_major)/num_thetas
#         prob=[p_minor]*num_thetas      
    
#     return generate_the_colors(rgb, theta, prob, design_name)



# def get_n_split_monochromatic_old(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
#                               keep_original_color = False, p0=1, use_orig_prob=True):    
#     n_step = 1 if n_step ==0 else n_step # this only returns the color complement        
#     theta = find_split_points(xx=perc_upper, split_step = n_step)              
#     if keep_original_color: theta.remove(0); theta.append(0) # moving zero to the end of list
#     else: theta.remove(0) 
#     num_thetas = len(theta) # has to be without the complement angle                
#     if use_orig_prob: 
#         prob = [p0]*(num_thetas)
#     else:
#         if keep_original_color: 
#             prob = [(1-p_major)/(num_thetas-1)]*(num_thetas-1) + [p_major]
#         else:
#             prob = [1/num_thetas]*num_thetas          
#     xxx = np.arange(0, 1, 1/(n_step+1))
#     return generate_the_colors(rgb, list(itertools.product(xxx, xxx)), prob, design_name='monochromatic', color_function=color_monochromatic)

