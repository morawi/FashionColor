# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 14:07:15 2021

@author: malrawi
"""
def get_59_class_names(): 
    # Item names ordered according to label id, 0 for background and 59 for wedges
    # A dictionary that contains each label name, and the possible (max) number of colors
    # The number of colors can be changed, these are used as upper bounds when estimating 
    # the number of colors in the item of the label
       
    min_ = 3  # min_num_colors
    low_ = 5  # low_num_colors
    mid_ = 8 # mid_num_colors
    max_ = 17  # max_num_colors
    
    
    class_names = {'background': 0,  # this will be ignored
                   'accessories': low_,  
                   'bag': mid_,  
                   'belt': min_,  
                   'blazer': max_,
                   'blouse': max_,  
                   'bodysuit': max_,  
                   'boots': mid_,  
                   'bra': mid_,  
                   'bracelet': low_,  
                   'cape': max_,  
                   'cardigan': max_,
                   'clogs': mid_, 
                   'coat': max_,  
                   'dress': max_, 
                   'earrings': min_, 
                   'flats': low_, 
                   'glasses': min_, 
                   'gloves': low_, 
                   'hair': mid_, # I had to add this up, because sometimes women's hair has a lot of colors 
                   'hat': mid_, 
                   'heels': min_, 
                   'hoodie': max_, 
                   'intimate': max_, 
                   'jacket': max_, 
                   'jeans': max_, 
                   'jumper': max_, 
                   'leggings': max_,
                   'loafers': low_, 
                   'necklace': min_, 
                   'panties':min_, 
                   'pants': max_, 
                   'pumps': low_, # there are multicolor pumps, but since this item will appear small in the image, it will be hard to get the colors
                   'purse': mid_, 
                   'ring': min_, 
                   'romper': max_,
                   'sandals': low_, 
                   'scarf': max_, 
                   'shirt': max_, 
                   'shoes': low_, 
                   'shorts': max_, 
                   'skin': min_,   # skin has one color
                   'skirt': max_, 
                   'sneakers': mid_,
                   'socks': low_, 
                   'stockings': mid_, 
                   'suit': max_, 
                   'sunglasses': min_, 
                   'sweater': max_,
                   'sweatshirt':max_, 
                   'swimwear': max_,
                   't-shirt': max_,
                   'tie': max_,
                   'tights': max_, 
                   'top': max_,
                   'vest': max_,
                   'wallet': min_, 
                   'watch': min_,
                   'wedges': mid_}
    
    return class_names
