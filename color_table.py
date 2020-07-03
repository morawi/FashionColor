# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:26:08 2020

@author: malrawi
"""

import pandas as pd


''' 
Indexing / Selection
The basics of indexing are as follows:

Operation	Syntax	Result
Select column	df[col]	Series
Select row by label	df.loc[label]	Series
Select row by integer location	df.iloc[loc]	Series
Slice rows	df[5:10]	DataFrame
Select rows by boolean vector	df[bool_vec]	DataFrame

'''

# colr_names = ['colr1', 'colr2', 'colr3', 'colr4']; 

class ColorTable():
    ''' Arrange the colors of fashion items/clothing into a pandas table'''
    def __init__(self, class_names, cnf):
        self.max_mumber_of_colors = cnf.max_num_colors
        fashion_items = ['im_name'] + class_names # add image name here        
        self.df = pd.DataFrame([(None,)* len(fashion_items)], columns = fashion_items,  index=[]) # Constructing the dataframe
        self.data_dict = {}
        # adding a trend item
        
        
    def add_fashion_item(self, item={'bag':[(40, 66, 93), (1, 2, 99)],'jacket' : (9, 0, 9) , 'pants' : (255,255, 255)}):
        self.df = self.df.append(item.item , ignore_index=True)        
        
        
    def __getitem__(self, index): # example, index = [0, 4], first and 5th rows; e.g.  x[[0,1]] where x is a color_table object
        return self.df.iloc[index]
    
    def __len__(self):
        return len(self.df)
    
    def get_cnt_color_as_df(self, all_rows_items, img_names):
        colr_names = ['colr_'+ str(i)  for i in range(self.max_mumber_of_colors)] # stores the colors of one item (e.g. skirt) for all clients
        pix_cnt_names = ['px_cnt_'+ str(i)  for i in range(self.max_mumber_of_colors)] # stores the pixel counts for one item(e.g. skirt) for all clients                    
        colr_df = pd.DataFrame([(None,)*len(colr_names)], columns = colr_names,  index=[]) # Constructing the dataframe
        pix_cnt_df = pd.DataFrame([(None,)*len(pix_cnt_names)], columns = pix_cnt_names,  index=[]) # Constructing the dataframe
        
        colr_df.index = colr_df.index.map(str)
        pix_cnt_df.index = pix_cnt_df.index.map(str)

        for jj, one_column_item in enumerate(all_rows_items): # '''idx has all clie key, key could be skirt, belt, dress, etc '''                        
            pix_cnt_color = dict(one_column_item) #  ''' each per_col_item has max_mumber_of_colrs  defined as pixel_count, followed by the color value '''
            pix_cnt = [*pix_cnt_color] # pix_cnt_color.keys() # these are pixel counts
            colors = [*pix_cnt_color.values()] # these are the colors corresponding to each pixel count
            
            pix_cnt = pix_cnt + [None] * (self.max_mumber_of_colors - len(pix_cnt)) # filling missing values wiht None if any
            pix_cnt_df.loc[len(pix_cnt_df)] = pix_cnt
            
            colors = colors + [None] * (self.max_mumber_of_colors - len(colors)) # filling missing values wiht None if any
            colr_df.loc[len(colr_df)] = colors
        colr_df.index = img_names
        pix_cnt_df.index = img_names
        ''' - To access one row, we use: colr_df.loc['0272.jpg'], where '0272.jpg' is the image name index
            - To acess one point at a specific row and column we can use, colr_df['colr_0']['0272.jpg']  '''
            
        return pix_cnt_df, colr_df

    
    def build_table(self): # arrange in    
        pix_cnt_df={}; colr_df ={} 
        for i, key in enumerate(self.df): # key here is fashion item, e.g. skirt, dress, etc                 
            idx = self.df[key].notna() # indices for all clients that are not None
            if key == 'im_name' or key == 'background' or not(idx.any()): continue                                    
            pix_cnt_df[key], colr_df[key] = self.get_cnt_color_as_df(self.df[key][idx], 
                                                                     self.df['im_name'][idx])
            
            ''' pix_cnt, colr, and img_names have one to one correspondence '''
        self.data_dict['pix_cnt_df'] = pix_cnt_df
        self.data_dict['colr_df'] = colr_df
        
            
             

# x = ColorTable()
# x.add_fashion_item(); x.add_fashion_item(); x.add_fashion_item()
# item = x[[0,1]]
# print(item)

# df = df.replace(r'^\s*$', np.nan, regex=True) # replacing None with Nan, if needed
# df = pd.DataFrame(students, columns = ['Jacket' , 'Pants', 'Blouse' , 'Dress'] ) # ,  index=['a', 'b'])