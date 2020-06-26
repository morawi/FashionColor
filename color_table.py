# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:26:08 2020

@author: malrawi
"""

import pandas as pd

def get_fashion_classes():
    class_names = ['accessories',  'bag',  'belt',  'blazer',
     'blouse',  'bodysuit',  'boots',  'bra',  'bracelet',  'cape',  'cardigan',
     'clogs', 'coat',  'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair',
     'hat', 'heels', 'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings',
     'loafers', 'necklace', 'panties', 'pants', 'pumps', 'purse', 'ring', 'romper',
     'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers',
     'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear',
     't-shirt', 'tie', 'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
    
    return class_names


class ColorTable():
    ''' Arrange the colors of fashion items/clothing into a pandas table'''
    def __init__(self):
        fashion_items = get_fashion_classes()
        fashion_trends = [(None,)*len(fashion_items)] # generating empty trend to be used for constructing the data frame
        self.df = pd.DataFrame(fashion_trends, columns = fashion_items,  index=[]) # Constructing the dataframe
        
        # adding a trend item
    def add_fashion_item(self, item={'bag':[(40, 66, 93), (1,2,99)],'jacket' : (9, 0, 9) , 'pants' : (255,255, 255)}):
        self.df = self.df.append( item , ignore_index=True)
        
        
    def __getitem__(self, index): # example, index = [0, 4], first and 5th rows; e.g.  x[[0,1]] where x is a color_table object
        return self.df.iloc[index]

# x = ColorTable()
# x.add_fashion_item(); x.add_fashion_item(); x.add_fashion_item()
# item = x[[0,1]]
# print(item)

# df = df.replace(r'^\s*$', np.nan, regex=True) # replacing None with Nan, if needed
# df = pd.DataFrame(students, columns = ['Jacket' , 'Pants', 'Blouse' , 'Dress'] ) # ,  index=['a', 'b'])