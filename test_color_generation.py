# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:43:25 2020

@author: malrawi
"""

import numpy as np
from generate_colors import *


min_rgb = 0; max_rgb = 256; rgb_val = np.random.randint(min_rgb, high=max_rgb, size=3, dtype = 'uint8')
#rgb_val  = [15, 76, 129]  # '''  Use panton clor values '''  15 76 129;  162 36 47 # Panton colors https://www.pantone.com/color-intelligence/fashion-color-trend-report/new-york-autumn-winter-2020-2021
rgb_val= [100, 140, 10]
n_split = 10
generate_all_col = False

if generate_all_col:
    color_pack = generate_pack_of_colors(rgb_val, n_split= n_split, match_mode = 'complement' )    
else:    
    fname= str(rgb_val)+ '-'+ str(n_split)+'-split'        
    design_pattern= get_n_split_monochromatic(rgb_val, 
                                              p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
                                              perc_upper=0.1, 
                                              n_step=n_split, 
                                              use_orig_prob=False,
                                              keep_original_color = True
                                              )
    color_pie_chart(design_pattern, fname=fname)
    print(n_split, 'Monochromatic')

    
    # design_pattern= get_n_split_analogous(rgb_val, 
    #                                           p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
    #                                           perc_upper=0.1, 
    #                                           n_step=n_split, 
    #                                           use_orig_prob=True,
    #                                           keep_original_color = True
    #                                           )
    # color_pie_chart(design_pattern, fname=fname)
    # print(n_split, 'Analogous')
        
    # design_pattern_opp = apply_opposite(design_pattern)
    # color_pie_chart(design_pattern_opp, fname='oppos_'+fname)
    # print(n_split, 'opposite on analogous')
    
    # design_pattern_shade = apply_shade(design_pattern_opp, shade_val=128)
    # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
    # print(n_split, 'shade on opposite')
    
    
    
    
    
    # design_pattern_comp= get_n_split_complementary(rgb_val, 
    #                                           p_major = 1/ (2*n_split+1), # this value only when analogous_bool=False
    #                                           n_step=n_split, 
    #                                           keep_original_color = True, 
    #                                           use_orig_prob=False)
    # color_pie_chart(design_pattern_comp, fname=fname)
    # print(n_split, 'Complement')
    
    # design_pattern_comp_opp = apply_opposite(design_pattern_comp)
    # color_pie_chart(design_pattern_comp_opp, fname=fname)
    # print(n_split, 'Opposite on Complement')
    
    # design_pattern_shade = apply_shade(design_pattern_comp, shade_val=128)
    # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
    # print(n_split, 'shade on complement')
    
    
