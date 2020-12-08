# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:53:57 2020

@author: malrawi
"""


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
import sys

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
        col_opposite = [255-rgb[0], 255-rgb[1], 255-rgb[2]]
        col_.append(col_opposite)        
    design_pattern['color'] = col_
    design_pattern['design_name']  = [design_pattern['design_name'][0] + '-opposite']
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    return design_pattern
    
    

def color_complementary(rgb, theta=0.5): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
        """returns RGB components of complementary color"""
        hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        rgb = hsv_to_rgb((hsv[0] + theta) % 1, hsv[1], hsv[2])
        return [int(rgb[0]), int(rgb[1]), int(rgb[2])]

def color_shade(rgb, shade=100): # when theta is 0.5, it will give color_complement(a, b, c), a value of theta =1 will gived the same color
        """returns RGB components of complementary color"""
        hsv = rgb_to_hsv(rgb[0], rgb[1], rgb[2])
        # rgb = hsv_to_rgb(hsv[0], (int(255*hsv[1] + shade)%255)/255, hsv[2] )
        rgb = hsv_to_rgb(hsv[0], hsv[1], (hsv[2] +shade)%255 )
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
    

def list_of_rgb_to_names(col_):
    color_names_obj = ColorNames()     
    hex_colors = [ RGB2HEX(col_val) for col_val in col_ ]
    names_of_colors = [color_names_obj.get_color_name(item) for item in  hex_colors]  
    return names_of_colors # , hex_colors

def generate_the_colors(rgb, theta, prob, design_name):
    col_=[]; 
    design_pattern = {}    
    for th in theta:
        col_.append(color_complementary(rgb, th) )
    design_pattern['color'] = col_ #  np.array(col_, dtype =float) 
    design_pattern['prob']  = prob # np.array(prob)
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    design_pattern['design_name'] = [design_name]
    return design_pattern


def get_n_split_complementary(rgb, perc_upper=None, n_step=0, p_major = 0.4, 
                              keep_original_color = False, 
                              p0=1, match_mode= 'complement'):
    
    if perc_upper==None:
        perc_upper = (9+n_step)/100 # the hue degree to depend on the number of splits, to allow stretching the colros if num of splits is high
    
    if n_step ==0: # this only returns the color complement 
        if match_mode =='complement':
            return generate_the_colors(rgb, theta=[0.5], prob=[p0], design_name= 'complement')
        else:
            n_step =1 # analogous does not work with n_step=0
           
    theta = find_split_points(xx=perc_upper, split_step = n_step)    
            
    # idx =match_mode.find('_');  match_mode_class = match_mode[:idx] if idx>0 else match_mode  # idx = -1 if '_' does not exist
    if match_mode =='complement':                                
        num_thetas = len(theta) # has to be without the complement angle        the
        theta = [thx+0.5 for thx in theta] # converting to complement           
        if keep_original_color: theta.append(0)
        else: num_thetas -=1        
        p_minor = (1-p_major)/num_thetas
        prob=[p_major]+ [p_minor]*num_thetas
    elif match_mode =='analogous': 
        if not keep_original_color: theta.remove(0)
        num_thetas = len(theta) # has to be without the complement angle                
        prob=[p0/num_thetas]*num_thetas
        
    prob = [p0*p for p in prob] # prob will no sum up to 1 in this case, as it depends on the priori p0
    return generate_the_colors(rgb, theta, prob, match_mode)


    
def find_split_points(xx=.15, split_step = 2):    
    xx = int(xx*100)
    xx = int(xx/split_step)*split_step # correcting nn so that it accepts division by split_step
    zz= np.array(range(-xx, 0, int (xx/split_step)) )/ 100
    z2= -np.flipud(zz)     
    return list(zz)+list([0])+list(z2)
        

def generate_pack_of_colors(rgb_val, n_split, p0=1, match_mode = 'complement', perc_upper=None):
    ''' 
    input - 
    match_mode
        - complement, complemen_opposite, complement_shade
        - analogous, analogous_opposite, analogous_shade, 
    rgb_val: 
        a single RGB value example (255, 121, 11)
    output 
        - resutl: dictionary of each color set
    
    '''     
    # fname= str(rgb_val)+ '-'+ str(n_split)+'-split' 
    
    color_pack = {}
    
    design_pattern_comp= get_n_split_complementary(rgb_val, 
                                              p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                                              n_step=n_split, 
                                              perc_upper = perc_upper,
                                              match_mode= 'complement', p0=p0); # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)
           
    design_pattern_analog = get_n_split_complementary(rgb_val, 
                                              p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                                              n_step=n_split, 
                                              perc_upper = perc_upper,
                                              match_mode= 'analogous', p0=p0); # print(n_split, 'Complement') ## color_pie_chart(design_pattern, fname=fname)

    
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