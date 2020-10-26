# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 12:26:08 2020

@author: malrawi

"""

import pandas as pd
import gc
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import colors
import matplotlib.pyplot as plt
import numpy as np
import pickle


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

# colr_names = ['colr1', 'colr2', 'colr3', 'colr4']

class ColorTable():
    ''' Arranges colors of fashion/clothing items into a pandas table. 
    Each outfit is stored into a single table. Tables of several outfits can be appended '''
    def __init__(self, class_names=None, max_num_colors=None, 
                 fashion_obj_name='season20-21'):
        if class_names != None:
            self.max_mumber_of_colors = max_num_colors
            fashion_items = ['im_name'] + class_names # add image name here        
            self.df = pd.DataFrame([(None,)* len(fashion_items)], 
                                   columns = fashion_items,  index=[]) # Constructing the dataframe
        self.data_dict = {}
        self.obj_name = fashion_obj_name
        # nself.num_entries = 0  # each person outfit has one entry / instance                 
                
    def save(self, fname, path = 'C:/MyPrograms/FashionColor/ColorFiles/'):
        '''  save data_dict to file  '''               
        
        with open(path+fname, 'wb') as fp:
            pickle.dump(self.data_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
        # with open('data.json', 'w') as fp:
        #     json.dump(data, fp)
                
        
    def load(self, fname, path = 'C:/MyPrograms/FashionColor/ColorFiles/'):   
        with open(path+fname, 'rb') as fp:
            self.data_dict = pickle.load(fp)
                      
    
    def update(self, color_object):
        self.data_dict.update(color_object.data_dict)
    
    def check(self):
        ''' TODO '''        
        return 10
    
    
    def append(self, item={'bag':[(40, 66, 93), (1, 2, 99)],'jacket' : (9, 0, 9) , 'pants' : (255,255, 255)}):
        '''the default item is some trivial input, used to show the user how to do use append '''
        self.df = self.df.append(item.item , ignore_index=True) 
        # self.num_entries +=1 # no need for this
    
        
    def __getitem__(self, index): # example, index = [0, 4], first and 5th rows; e.g.  x[[0,1]] where x is a color_table object
        # return self.df.iloc[index]
        return self.data_dict
    
    # def __len__(self): # the length is not fixed, sometimes we have 1 pants, 2 shirts 
    #     return # self.num_entries
    
    def get_cnt_color_as_df(self, all_rows_items, img_names):
        colr_names = ['color_'+ str(i)  for i in range(self.max_mumber_of_colors)] # stores the colors of one item (e.g. skirt) for all clients
        pix_cnt_names = ['pxl_prob_'+ str(i)  for i in range(self.max_mumber_of_colors)] # stores the pixel counts for one item(e.g. skirt) for all clients                    
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

        
    def build_table(self): 
        ''' This function builds a table so that colors are stored in a DataFraem, 
            done for each clothing item ''' 
    
        if self.df.empty:        
            print('Error: There are no data to fill the table. Use append() \
                   to add data.')                   
            return 0
         
        pix_cnt_df={}; colr_df ={}        
        for i, key in enumerate(self.df): # key here is fashion item, e.g. skirt, dress, etc                 
            idx = self.df[key].notna() # indices for all clients that are not None
            if key == 'im_name' or key == 'background' or not(idx.any()): continue                                    
            pix_cnt_df[self.df['im_name'][idx]], colr_df[self.df['im_name'][idx]] = self.get_cnt_color_as_df(self.df[key][idx], key)
            
            ''' pix_cnt, colr, and img_names have one to one correspondence '''
        self.data_dict['pix_cnt_df'] = pix_cnt_df
        self.data_dict['colr_df'] = colr_df
        self.data_dict['obj_name'] = self.obj_name
                       
        print('Calling build_table(). Warning: This will build the table based on each item.\nThe original DataFrame containing the information will be cleared to save memory.')
        self.df = self.df[0:0] # Now, let's free the memory used by df
        gc.collect() # garbage collection after deletion
        
    def build_table_old(self): 
        ''' This function builds a table so that colors are stored in a DataFraem, 
            done for each clothing item ''' 
    
        if self.df.empty:        
            print('Error: There are no data to fill the table. Use append() \
                   to add data.')                   
            return 0
         
        pix_cnt_df={}; colr_df ={}        
        for i, key in enumerate(self.df): # key here is fashion item, e.g. skirt, dress, etc                 
            idx = self.df[key].notna() # indices for all clients that are not None
            if key == 'im_name' or key == 'background' or not(idx.any()): continue                                    
            pix_cnt_df[key], colr_df[key] = self.get_cnt_color_as_df(self.df[key][idx], 
                                                                     self.df['im_name'][idx])
            
            ''' pix_cnt, colr, and img_names have one to one correspondence '''
        self.data_dict['pix_cnt_df'] = pix_cnt_df
        self.data_dict['colr_df'] = colr_df
        self.data_dict['obj_name'] = self.obj_name
                       
        print('Calling build_table(). Warning: This will build the table based on each item.\nThe original DataFrame containing the information will be cleared to save memory.')
        self.df = self.df[0:0] # Now, let's free the memory used by df
        gc.collect() # garbage collection after deletion    
                    
    def analyze(self):
        ''' This function can be used to build the color distributions of each 
        clothing item ... This function should be called after calling build_table() '''
        if self.data_dict=={}:
            print('you need to run build_table() to generate the data used in the analysis')
            return 0
        
        # color analysis
        L1_clr_nm = 'colr_df'  # color group 
        L1_cnt_nm = 'pix_cnt_df'  # pixel count group ... both group have one to one correspondence
        for L2_item_nm in  self.data_dict[L1_clr_nm].keys(): # L2_KEYS          
            g_clr = self.data_dict[L1_clr_nm][L2_item_nm].columns
            g_cnt = self.data_dict[L1_cnt_nm][L2_item_nm].columns
            for L3_clr_nm, L3_cnt_nm  in zip(g_clr, g_cnt) : # L3_COLUMN_KEYS 
                colr =  self.data_dict[L1_clr_nm][L2_item_nm][L3_clr_nm]
                pix_cnt = self.data_dict[L1_cnt_nm][L2_item_nm][L3_cnt_nm]
                self.plot_table(colr, pix_cnt, L2_item_nm +'.'+ L3_clr_nm)
                
            
    def plot_table(self, pixel_colors, pix_cnt, item_name):  
        
        pix_scale = 50
                
        # removing NaN values
        pix_cnt = np.array(list(pix_cnt[pix_cnt.notna()]))
        if pix_cnt.size==0: print('Item ', item_name, ' does not exist'  ); return                    
        pixel_colors = np.array( list( pixel_colors[pixel_colors.notna()] ))       
                
        ch1, ch2, ch3 = np.hsplit(pixel_colors, 3)
        ch1.reshape(1,-1)[0]
        ch2.reshape(1,-1)[0]
        ch3.reshape(1,-1)[0]
        norm = colors.Normalize(vmin=-1.,vmax=1.)
        norm.autoscale(pixel_colors)
        pixel_colors = norm(pixel_colors).tolist()
        # ch1=np.array(ch1.round(), dtype='int')
        # ch2=np.array(ch2.round(), dtype='int')
        # ch3=np.array(ch3.round(), dtype='int')
        
        marker_size = (pix_scale * pix_cnt).round() # [ 0.15  for n in range(len(ch1))] # the size of the marker, we can use the pixel_cnt or the probability of every piont in the (ch1,ch2,ch3)
           
        #plotting         
        fig = plt.figure()        
        mm = str( (100*np.mean(pix_cnt) ).round()/100) 
        ss = str( (100*np.std(pix_cnt) ).round()/100)        
        title = str(len(pix_cnt)) + ' '+item_name + ' '  r'$('+mm+ '\pm'+ss+ ')$'
        fig.suptitle(title, fontsize=20)
        axis = Axes3D(fig)
        axis.tick_params(axis='x', labelsize=16)
        axis.tick_params(axis='y', labelsize=16)
        axis.tick_params(axis='z', labelsize=16)
        axis.scatter(ch1, ch2, ch3, facecolors=pixel_colors, marker=".", s = marker_size)    
        axis.set_xlabel("Hue", fontsize=18, labelpad=15)
        axis.set_ylabel("Saturation", fontsize=18, labelpad=15)
        axis.set_zlabel("Value", fontsize=18, labelpad=7, position=(-2,-2))
        N=4                
        axis.set_xticks(np.round(np.linspace(0, 255, N), ))
        axis.set_yticks(np.round(np.linspace(0, 255, N), ))
        axis.set_zticks(np.round(np.linspace(0, 255, N), ))
        
        plt.savefig('./Figures/' + item_name +".pdf", bbox_inches='tight')
        plt.show()
        
        '''
        shape of  pixel_colors
        array([[  0,   0, 255],
       [  0,   0, 255],
       [  0,   0, 255],
       ...,
       [  0,   0, 220],
       [  0,   0, 221],
       [  0,   0, 221]], dtype=uint8)
        
        '''
               
            
            

# x = ColorTable()
# x.append(); x.append(); x.append()
# item = x[[0,1]]
# print(item)

# df = df.replace(r'^\s*$', np.nan, regex=True) # replacing None with Nan, if needed
# df = pd.DataFrame(students, columns = ['Jacket' , 'Pants', 'Blouse' , 'Dress'] ) # ,  index=['a', 'b'])