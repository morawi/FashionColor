# -*- coding: utf-8 -*-
"""
Created on Thu Nov 12 12:43:25 2020

@author: malrawi
"""

import numpy as np
from generate_colors import *


min_rgb = 0; max_rgb = 256; rgb_val = np.random.randint(min_rgb, high=max_rgb, size=3, dtype = 'uint8')
#rgb_val  = [15, 76, 129]  # '''  Use panton clor values '''  15 76 129;  162 36 47 # Panton colors https://www.pantone.com/color-intelligence/fashion-color-trend-report/new-york-autumn-winter-2020-2021
rgb_val= [135, 52, 70]
# rgb_val= [110, 114, 115]
# rgb_val= [180, 128, 50]
n_split = 10
generate_all_col = False

fname= str(rgb_val)+ '-'+ str(n_split)+'-split'  

if generate_all_col:
    color_pack = generate_pack_of_colors(rgb_val, n_split= n_split, match_mode = 'complement' )    
    
else:    
    
    design_pattern_comp= get_n_split_complementary(rgb_val,   
                                              perc_upper=0.1, 
                                              n_step=n_split, 
                                              keep_original_color = True, 
                                              use_orig_prob=False)
    color_pie_chart(design_pattern_comp, fname=fname)
    print(n_split, 'Complement')
    
      
    design_pattern_purity= get_n_split_purity(rgb_val, 
                                              perc_upper=0.5, 
                                              n_step=n_split, 
                                              use_orig_prob=False,
                                              keep_original_color = True
                                              )
    color_pie_chart(design_pattern_purity, fname=fname)
    print(n_split, 'Purity')

    fname= str(rgb_val)+ '-'+ str(n_split)+'-split'        
    design_pattern= get_n_split_monochromatic(rgb_val, 
                                              perc_upper=1, #  value 1 will generate up to 255, lower values will generate dark colors up to perc_upper*255
                                              n_step=n_split, 
                                              use_orig_prob=False,
                                              keep_original_color = True
                                              )
    color_pie_chart(design_pattern, fname=fname)
    print(n_split, 'Monochromatic')

    
    design_pattern= get_n_split_analogous(rgb_val, 
                                        perc_upper=0.1, 
                                        n_step=n_split, 
                                        use_orig_prob=False,
                                        keep_original_color = True
                                              )
    color_pie_chart(design_pattern, fname=fname)
    print(n_split, 'Analogous')
        
    # design_pattern_opp = apply_opposite(design_pattern)
    # color_pie_chart(design_pattern_opp, fname='oppos_'+fname)
    # print(n_split, 'opposite on analogous')
    
    # design_pattern_shade = apply_shade(design_pattern_opp, shade_val=128)
    # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
    # print(n_split, 'shade on opposite')
    
    
    
    
    
    
    # design_pattern_comp_opp = apply_opposite(design_pattern_comp)
    # color_pie_chart(design_pattern_comp_opp, fname=fname)
    # print(n_split, 'Opposite on Complement')
    
    # design_pattern_shade = apply_shade(design_pattern_comp, shade_val=128)
    # color_pie_chart(design_pattern_shade, fname='shade_'+fname)
    # print(n_split, 'shade on complement')
    
    
