# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 14:53:57 2020

@author: malrawi
"""


# Complementary Colours Algorithm - www.101computing.net/complementary-colours-algorithm
# on color Complementary vs color opposite https://www.quora.com/What-is-the-complementary-colour-of-black
# color opposite, althoiug callled complementary https://www.101computing.net/complementary-colours-algorithm/
# color complement code   https://stackoverflow.com/questions/40233986/python-is-there-a-function-or-formula-to-find-the-complementary-colour-of-a-rgb
# there is a nice hue wheel that shows how much color changes with the angle

#Complete the code here...
from colorsys import rgb_to_hsv, hsv_to_rgb
import numpy as np
from color_utils import RGB2HEX
from color_names import ColorNames
import matplotlib.pyplot as plt

def color_pie_chart(design_pattern, figure_size=(9, 6), fname=None):  
        color_names_obj = ColorNames()    
        pixel_counts, hex_colors = zip(*[( design_pattern['prob'][i], RGB2HEX(design_pattern['color'][i])) for i in range(len(design_pattern['prob'])) ])      
        # names_of_colors = [color_names_obj.get_color_name(item) for item in  hex_colors]            
        pixel_counts_labels =[ str(int(np.round(100*item, 2)))+'%' for item in pixel_counts]   # color names from color_names class can be added here
        # print(names_of_colors)
        plt.figure(figsize = figure_size)
        # plt.suptitle(label_val, fontsize=22)
        plt.pie(100*np.array(pixel_counts), labels = pixel_counts_labels, colors = hex_colors,
                rotatelabels = False, textprops={'fontsize': 18})            
        # plt.pie(num_pixel_percent, labels = hex_colors, colors = hex_colors, # drawing color vlaues in hexdecimal
        #         rotatelabels = False)            
        plt.show()                               
        if fname:    
            plt.savefig('./Figures/' + fname + '.png')        
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
    design_pattern['design_name']  = design_pattern['design_name']  + 'opposite'
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
    design_pattern['design_name']  = design_pattern['design_name']  + 'shade'
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    return design_pattern

    
    

def list_of_rgb_to_names(col_):
    color_names_obj = ColorNames()     
    hex_colors = [ RGB2HEX(col_val) for col_val in col_ ]
    names_of_colors = [color_names_obj.get_color_name(item) for item in  hex_colors]  
    return names_of_colors, hex_colors

def generate_colors(rgb, theta, prob, design_name):
    col_=[]; 
    design_pattern = {}    
    for th in theta:
        col_.append(color_complementary(rgb, th) )
    design_pattern['color'] = col_
    design_pattern['prob']  = prob
    design_pattern['design_name'] = design_name     
    design_pattern['color_names'] = list_of_rgb_to_names(col_)
    return design_pattern





def get_n_split_complementary(rgb, perc_upper, n_step, p_major = 0.4, 
                              analogous=False, design_name='something'): # theta = []
    if n_step ==0:
        return generate_colors(rgb, theta=[0, 0.5], prob=[0.5, 0.5], design_name= design_name)
    theta = find_split_values(xx=perc_upper, split_step = n_step)
    if not analogous:
        theta.remove(0)
        num_thetas = len(theta) # has to be without the complement angle        
        theta =[0.5]+theta # adding the complement angle            
        p_minor = (1-p_major)/num_thetas
        prob=[p_major]+ [p_minor]*num_thetas
    else:
        num_thetas = len(theta) # has to be without the complement angle        
        p_minor = (1-p_major)/num_thetas
        prob=[p_minor]*num_thetas      
       
    print(theta)
    return generate_colors(rgb, theta, prob, design_name)
    
    
def find_split_values(xx=.15, split_step = 2):    
    xx = int(xx*100)
    xx = int(xx/split_step)*split_step # correcting nn so that it accepts division by split_step
    zz= np.array(range(-xx, 0, int (xx/split_step)) )/ 100
    z2= -np.flipud(zz)     
    return list(zz)+list([0])+list(z2)
        
'''  Use panton clor values '''

min_rgb = 220
max_rgb = 255
rgb_val = np.random.randint(min_rgb, high=max_rgb, size=3, dtype = 'uint8')
# rgb_val  = [15, 76, 129]  #  15 76 129;  162 36 47 # Panton colors https://www.pantone.com/color-intelligence/fashion-color-trend-report/new-york-autumn-winter-2020-2021



n_split = 1
analogous_bool=False # when this is True, we get the analogous and not the complement
design_pattern= get_n_split_complementary(rgb_val, 
                                          p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                                          perc_upper=0.2, 
                                          n_step=n_split, 
                                          analogous=analogous_bool)
color_pie_chart(design_pattern)

design_pattern_opp = apply_opposite(design_pattern)
color_pie_chart(design_pattern_opp)
print(n_split, 'opposite')

design_pattern_shade = apply_shade(design_pattern_opp, shade_val=128)
color_pie_chart(design_pattern_shade)
print(n_split, 'shade')




# def color_complement(rgb):
#     a, b, c=rgb[0], rgb[1], rgb[2]
#     if c < b: b, c = c, b
#     if b < a: a, b = b, a
#     if c < b: b, c = c, b
#     k=a+c
#     return [k - u for u in rgb]    
# print(color_complement(design_pattern['color'][0]  ))


# design_pattern = get_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('Complementary')


# design_pattern = get_split_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('1-split Complementary')


# design_pattern = get_double_split_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('2-split Complementary')


# design_pattern = get_triple_split_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('3-split Complementary')


# design_pattern = get_analogous(rgb_val)
# color_pie_chart(design_pattern)
# print('1-split analogous')


# design_pattern = get_double_analogous(rgb_val)
# color_pie_chart(design_pattern)
# print('2-split analogous')

# design_pattern = get_triple_analogous(rgb_val)
# color_pie_chart(design_pattern)
# print('3-split analogous')


# design_pattern = get_opposite(design_pattern))
# color_pie_chart(design_pattern)
# print('3-split analogous opposite')

# design_pattern = get_double_split_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('2-split Complementary')

# design_pattern = get_split_complementary(rgb_val)
# color_pie_chart(design_pattern)
# print('Complementary')


# def get_complementary(rgb, theta=[0, 0.5], prob=[0.5, 0.5], design_name='something'):
#     return generate_colors(rgb, theta, prob, design_name)

# def get_split_complementary(rgb, theta=[0.5, -0.1, 0.1], prob=[0.4, 0.3, 0.3], design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)

# def get_double_split_complementary(rgb, theta=[.5, -.2, -.1, .1, .2], prob=[0.4, 0.15, 0.15, 0.15, 0.15], design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)

# def get_triple_split_complementary(rgb, theta=[.5, -.2, -.15, -.1, .1, .15, .2], prob=[0.4]+[0.1]*6, design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)



# def get_analogous(rgb, theta=[-.1, 0, .1], prob=[1/3]*3, design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)

# def get_double_analogous(rgb, theta=[-0.2, -.1, 0, .1, .2], prob=[1/5]*5, design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)

# def get_triple_analogous(rgb, theta=[-.15, -.1, -.05, 0, .05, .1, .15], prob=[1/7]*7, design_name='something'): # theta = []
#     return generate_colors(rgb, theta, prob, design_name)
    