# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 18:15:23 2020

@author: malrawi


"""
from color_table import ColorTable
import pickle
import numpy as np
from distance_troch import euclidean_distance # , cosine_similarity_n_space
import itertools
from generate_colors import generate_pack_of_colors
import copy

from colorsys import rgb_to_hsv

class ColorMatching():
    ''' Arranges colors of fashion/clothing items into a pandas table. 
    Each outfit is stored into a single table. Tables of several outfits can be appended '''
    def __init__(self):
        
        self.ref_data_dict = None                  
   

    def load(self, fname, path = 'C:/MyPrograms/FashionColor/ColorFiles/'):   
        with open(path+fname, 'rb') as fp:
            self.ref_data_dict = pickle.load(fp)
   
    def all_query_items_exist(self, items_to_match, query_items): 
        ''' Returns False if not all items to match are in ref_items, else returns True '''
                
        return set(items_to_match).issubset(query_items)
            
            
    
    def get_image_names(self, obj): # example: clothing_item = 'jacket'       
        return list(obj.data_dict['colr_df'].keys())
    
    
    
    def to_2D_mat(self, obj, img_name, item, dict_name):
        'Converts the color to a 2D matrix; output is RGB x NoColros'
        c_vals = obj.data_dict[dict_name][img_name].loc[item].values                 
        c_vals = np.reshape(c_vals, (c_vals.shape[0], -1))
        c_vals = np.vstack(c_vals[:, 0]).astype(np.float) # now, torch.tensor(aa) is OK
        return c_vals
    
    def remove_nans_from_numpy(self, inp):
        ''' Removing nan's from a numpy array, will always return a 1D array 
        even if the input is 2D or 3D '''
        return inp[~np.isnan(inp)] # return inp[~np.isnan(inp).any(axis=1)] # this keeps the 2D structure
        
    def remove_2D_nans_from_numpy(self, inp):
        ''' Removing nan's from a numpy array, will always return a 1D array 
        even if the input is 2D or 3D '''
        if np.isnan(inp).any(): 
            inp = inp[~np.isnan(inp).any(axis=1)]
            
        return inp  # this keeps the 2D structure
        

    
    def sort_and_store_record(self, dist_metric, catalogue_name, num_best_maches):
        ' Stores and sorts the results using the distance values'
        result = {}     
        result['dist_metric'] = dist_metric
        result['catalogue_name'] = catalogue_name 
        result['info'] = [] # to be appended with info
                                
        # sorting records according to distance                           
        idx_sorted_dist = np.argsort(result['dist_metric'])        
        result['catalogue_name'] = [result['catalogue_name'][i] for i in idx_sorted_dist]
        result['dist_metric'] = [result['dist_metric'][i] for i in idx_sorted_dist]
        
        # selecting the num of best matches
        result['dist_metric'] = result['dist_metric'][:num_best_maches]
        result['catalogue_name'] = result['catalogue_name'][:num_best_maches]   
        self.remove_dublicates(result) # removal of duplicates has to be done after the soreting, as we don't want to remove low distacne matches
        return result 
   
       
        
    def rgb2hsv(self, rgb_val):
        lambda_val = 2 # to give more weight on the hue in the distance 
        
        hsv_out = []
        for rgb in rgb_val:
            # print(rgb)
            hh = list(rgb_to_hsv(rgb[0],rgb[1], rgb[2]))            
            hh[2] = (hh[2]/255)/lambda_val  # reducing the saturation, to give more weight to the hue
            hh[1] = hh[1]/lambda_val   # reducting the value            
            hsv_out.append(hh)               
            # print('-----', '\n', hh, rgb_val)
        return hsv_out
        
                
    def find_e_distance(self, c_ref, p_ref, c_query, p_query):
        ''' $$e_{ij}(\text{s(group)}, \text{ground_truth}) = \frac{|p_i*P_j| |a_i - A_j|}{(p_i+P_j)^n}
            
            e_{ij}(\text{s(group)}, \text{ground_truth}) = \frac{|p_i*P_j| |a_i - A_j|}{(p_i+P_j)^n}
            $$            
            # if we want to test with 0s instead of nans
            # c_ref = np.nan_to_num(c_ref, 0)
                 
        '''
        
        # we need to make sure c_ref and c_query are of the same size, do embedding otherwise  
        
        # rgb_out = rgb_in.copy()
        # rgb_out.fill(np.nan)
        
        ''' IMPORTANT
        We may not need nans or 0 padded AT ALLL
        
        '''
        use_hsv = True
        
        if use_hsv:            
            c_query = self.remove_2D_nans_from_numpy(c_query)                        
            c_ref = self.remove_2D_nans_from_numpy(c_ref)
            c_query = self.rgb2hsv(c_query)
            c_ref = self.rgb2hsv(c_ref)  
                    
        dist_clr = euclidean_distance(np.array(c_ref, dtype=float), np.array(c_query, dtype=float))
        dist_clr = self.remove_nans_from_numpy(dist_clr)        
        
        prob_mult  = (np.array(p_ref, dtype=float).reshape(-1,1) * np.array(p_query, dtype=float)) 
        prob_mult = self.remove_nans_from_numpy(prob_mult) 

        
        return np.mean(dist_clr*prob_mult) # this is problematic, if the idstance of probability is zero, and the color is wrong, it will give 0 ...meaning perfecct color match, which is wrong



        
    def generage_item_pairs(self, wardrobe_obj, items_to_match):
        ''' Returns all the items as pairs, or triplets, depending on the number of items in
        items_to_match. For example, if we have "shirt" and "pants", then, 
        this function returns all the images of the wardrobe that has "shirt" and "pants", like
        ("shirt1", "pants1"  ),("shirt2", "pants2"  ), and so on  '''
        query_img_names = []        
        for item in items_to_match:
                query_img_names.append(wardrobe_obj.data_dict['pix_cnt_df'][item].index.tolist())
        if len(query_img_names)==1:
            return query_img_names[0] # this is the case when there is only one-item-to-match, this case errupts when matching wardrobe to itself
        if len(query_img_names)==2:
            item_pairs = list(itertools.product(query_img_names[0], query_img_names[1]))
        elif len(query_img_names)==3:
            item_pairs = list(itertools.product(query_img_names[0], query_img_names[1], query_img_names[2] ))
        elif len(query_img_names)==4:
            item_pairs = list(itertools.product(query_img_names[0], query_img_names[1], query_img_names[2], query_img_names[3] ))
        elif len(query_img_names)==5:
            item_pairs = list(itertools.product(query_img_names[0], query_img_names[1], query_img_names[2], query_img_names[3], query_img_names[4] ))
        elif len(query_img_names)==6:
            item_pairs = list(itertools.product(query_img_names[0], query_img_names[1], query_img_names[2], query_img_names[3], query_img_names[4], query_img_names[5]  ))
        elif len(query_img_names)>6:
           print('TODO ... need to add an item pair generation for more than 5 items to match')
           exit()
           
        # print('NOTE! There are ', len(item_pairs), 'compbinations for', items_to_match)
        return item_pairs
       
        
    def remove_dublicates(self, result): 
        '''         
         Removal of duplicates has to be done after the soreting, as we don't want to remove low distacne matches
         Two, or more, might be identified as matches, and we dont want that  
         
        '''
        z = result['catalogue_name']
        dublicate_idx = []
        for i in range(len(z)):
            for j in range(i+1, len(z)):
                if sorted(z[i]['matched_items'])==sorted(z[j]['matched_items']): # sorted is needed as they might not be in order                 
                    dublicate_idx.append(j)
                    i=i-1
                    break
        cnt = 0            
        for j in sorted(dublicate_idx):
            result['catalogue_name'].pop(j-cnt)
            result['dist_metric'].pop(j-cnt)            
            cnt +=1            
            # print(j)
        if cnt>0:
            print()
            result['info'].append('dublicate matches removed')
            
            
                    
    def ref_v_ref_matches(self, ref_obj, query_obj, items_to_match = ['jacket', 'pants', 'shirt' ], 
               num_best_maches = 10): 
        '''  query_obj should contain clothing from whole person (outfit) and we find the best match 
        after comparing it to other outfits stored in ref_obj (which is something like the trend catalogue) 
        
                
        output
        - result: a dictionary that contains the matches and the distances '''        
        
        items_to_match = [items_to_match] if not isinstance(items_to_match, list) else items_to_match
        print('matching according to', ref_obj.data_dict['obj_name'], 'catalogue' )              
        
        ref_images = self.get_image_names(ref_obj)
        query_images = self.get_image_names(query_obj)
        
         
        dist_metric = []
        catalogue_name = []
        
        for ref_img_name in ref_images:
            ref_items  = list(ref_obj.data_dict['pix_cnt_df'][ref_img_name].index)
            if not self.all_query_items_exist(items_to_match, ref_items): continue
            for query_img_name in query_images:
                query_items  = list(query_obj.data_dict['pix_cnt_df'][query_img_name].index)
                if not self.all_query_items_exist(items_to_match, query_items): continue
                
                dist_item = 0
                for item in items_to_match:                 
                    p_ref = ref_obj.data_dict['pix_cnt_df'][ref_img_name].loc[item].values            
                    c_ref = self.to_2D_mat(ref_obj, ref_img_name, item, 'colr_df')
                    
                    p_query = query_obj.data_dict['pix_cnt_df'][query_img_name].loc[item].values            
                    c_query = self.to_2D_mat(query_obj, query_img_name, item, 'colr_df')
                    
                    
                    dist_item += self.find_e_distance(c_ref, p_ref, c_query,p_query)
                
                dist_metric.append(dist_item) # much easier to store the distance separately, as we need it to find the min values to obtain the best matches
                catalogue_name.append({'query':query_img_name, 'reference': ref_img_name, 'matched_items': items_to_match}) # registering the catalogue that possibly matches the query
        
        result = self.sort_and_store_record(dist_metric, catalogue_name, num_best_maches)
        result['info'].append(ref_obj.data_dict['obj_name'])
        return result

        
    
    def user_wardrobe_v_ref_matches(self, ref_obj, wardrobe_obj, items_to_match = ['jacket', 'pants', 'shirt' ], 
                   num_best_maches = 10): 
            '''  query_obj should contain clothing from  one person's wardrob and we find the best match after comparing it
            to the ref_obj (which is something like the trend catalogue) 
            
                        
            output
            - result: a dictionary that contains the matches and the distances 
            
            What is a wordrobe? A wardrobe is a collection of single-item images,
            that is, eacah image has just one single item. These collections
            are stored in a folder named by the user_name            '''        
            
            self.wardrobe_sanity_check(wardrobe_obj, items_to_match)
                                   
            dist_metric = [];  catalogue_name = []
            print('matching according to', ref_obj.data_dict['obj_name'], 'catalogue' )                        
            item_pairs = self.generage_item_pairs(wardrobe_obj, items_to_match)              
                
            for ref_img_name in self.get_image_names(ref_obj):
                ref_items  = list(ref_obj.data_dict['pix_cnt_df'][ref_img_name].index) # this does not work in debug mode, but ok in run mode
                ''' we need to check of items_to_match in wardrobe_iems '''
                if not self.all_query_items_exist(items_to_match, ref_items): continue
                print('Image with', items_to_match, 'is:', ref_img_name)  
                                           
                for item_pair in item_pairs:
                    dist_item = 0
                    for query_img_name in item_pair:
                        item = query_img_name[:query_img_name.find('_')]                         
                        p_ref = ref_obj.data_dict['pix_cnt_df'][ref_img_name].loc[item].values
                        c_ref = self.to_2D_mat(ref_obj, ref_img_name, item, 'colr_df')
                        p_query = wardrobe_obj.data_dict['pix_cnt_df'][item].loc[query_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
                        c_query = self.to_2D_mat(wardrobe_obj, item, query_img_name, 'colr_df')

                        dist_item += self.find_e_distance(c_ref, p_ref, c_query, p_query)                        
                                                
                    dist_metric.append(dist_item) # much easier to store the distance separately, as we need it to find the min values to obtain the best matches
                    catalogue_name.append({'matched_items': item_pair, 'query':items_to_match,  'reference': ref_img_name }) # registering the catalogue that possibly matches the query
                
            result  = self.sort_and_store_record(dist_metric,catalogue_name, num_best_maches)
            result['info'].append(ref_obj.data_dict['obj_name'])
            
            
            return result

    
    def wardrobe_sanity_check(self, wardrobe_obj, items_to_match):
        items_to_match = [items_to_match] if not isinstance(items_to_match, list) else items_to_match # if the input is not a list, make it a list
        if len(items_to_match)<2: 
            print('Error: there is less than 2 items to match, min items should be 2')
            exit()
       
        wardrobe_items = self.get_image_names(wardrobe_obj) # here, we get the labels and not the image names, since we've restructured the wardrobe
        if not self.all_query_items_exist(items_to_match, wardrobe_items):                           
            print('Wardrobe does not have all items to match, these are:', items_to_match)
            print('Wardrobe, on the other hand, has only these labels:', wardrobe_items)
            exit()            
        return True
    
    def combine_packs(self, color_pack):  
        '''' Restructuring the color_pack so that multi-color items are correctly organized '''
        output_pack = copy.deepcopy(color_pack)[0]
        
        for record in color_pack[1:]:            
            for key1 in record.keys():                  
                for key2 in record[key1].keys():
                    value = record[key1][key2] # as it is a single item list, we need to remove the list, take the first element only
                    output_pack[key1][key2]= output_pack[key1][key2] + value                                        
        return output_pack
    
    
    
    
    def get_color_pack(self, rgb_in, p_in, n_split, match_mode = 'complement'):

        '''     now, it is possible that c_1_orig has more than one RGB value 
                we need to deal with it. Two approaches:
                    1- use n_split>0 for each RGB, and consider each RGB separately
                    2- use low n_split (0,1) and merge both RGBs into one structure            
        '''
        color_pack = []
        for i, rgb_val in enumerate(rgb_in):
            if np.isnan(rgb_val).any(): break            
            record = generate_pack_of_colors(rgb_val, n_split= n_split, p0=p_in[i], match_mode = match_mode )
            color_pack.append( record )   
        
        return self.combine_packs(color_pack) 
    
    
    
    
    def get_split_point(self, use_random_split, num_of_colors, n_split ):
        ''' generate a new splitting point depeding on the number of colors,  '''
        if use_random_split:                                
            n_split = np.random.randint(0, num_of_colors)
        return n_split
        
    
    
    # this is not in use now
    def user_wardrobe_vs_itself_matches_multi(self, wardrobe_obj, items_to_match = ['jacket', 'pants', 'shirt' ], 
                  num_best_maches = 10, n_split=1, match_mode = 'complement', use_random_split = False):
        '''match_mode: complement, analogous, mix (has analogous, complement, opposite, and shade) 
        num_best_maches: the result may be lower than the selected after removing duplicates 
        n_split: used to split complement and analogous, higher value result in more colors
        n_split=0 only works with complement, it should be >=1 for analogous
        use_random_split: if True, n_split will be generated randomly depending on the ref item, and used 
        to find complement colors splitted on the query '''
        
        dist_metric = []; catalogue_name = []
        self.wardrobe_sanity_check(wardrobe_obj, items_to_match)
        
        for item_ref in items_to_match: # this mainly has one item, but did a for-loop for generality            
            for ref_img_name in wardrobe_obj.data_dict['colr_df'][item_ref].index:
                p_ref = wardrobe_obj.data_dict['pix_cnt_df'][item_ref].loc[ref_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
                c_ref = self.to_2D_mat(wardrobe_obj, item_ref, ref_img_name, 'colr_df')   
                n_split = self.get_split_point(use_random_split, len(self.remove_nans_from_numpy(p_ref)), n_split )
                items_to_match_minus_ref = items_to_match.copy()
                items_to_match_minus_ref.remove(item_ref) # we are comparing each piece in item with the rest of items in items_to_match            
                item_pairs = self.generage_item_pairs(wardrobe_obj, items_to_match_minus_ref)
                # generate the colors of every query_image_name, then, find the distance with each single item
                for items_query in item_pairs:
                    dist_item = 0; comp_conunt = 0
                    if not isinstance(items_query, list): items_query=[items_query] # make it a list if it has only one item
                    for query_img_name in items_query:                        
                        item_query_name = query_img_name[:query_img_name.find('_')]
                        p_query_orig = wardrobe_obj.data_dict['pix_cnt_df'][item_query_name].loc[query_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
                        c_query_orig = self.to_2D_mat(wardrobe_obj, item_query_name, query_img_name, 'colr_df')
                        # now, we have to find the color complement of c_query and compare it with the ref
                        query_color_pack = self.get_color_pack(c_query_orig, 
                                                               p_query_orig, n_split, match_mode = match_mode) # we have to take the effect of the original probability and multiply it by the new one from the split, necessary when c_query_orig has more than one RGB with different probabilities
                                                
                        for c_query_key in query_color_pack.keys():
                            ''' we need a distance for each pack, one for complement, one for analogous, etc... 
                                so, no need for the for-loop here '''
                            if c_query_key == 'num_split': continue # this is not key, but split num / information                            
                            c_query_instance = query_color_pack[c_query_key]['color']                            
                            p_query_instance = query_color_pack[c_query_key]['prob'] # this is the resutlant split probabilit multiplied by the prior p0=p_query_orig, that is, low values p_query_orig should penalize the new colors                                    
                            comp_conunt = comp_conunt +1
                            dist_item += self.find_e_distance(c_query_instance, 
                                                              p_query_instance, 
                                                              c_ref, p_ref)
                    dist_metric.append(dist_item/comp_conunt) # much easier to store the distance separately, as we need it to find the min values to obtain the best matches
                    catalogue_name.append({ 'matched_items': sorted( list(items_query) + [ref_img_name]) }) # registering the catalogue that possibly matches the query
                                     
                        
        print('------ There are ', comp_conunt*len(item_pairs)*len(ref_img_name)*len(items_to_match),
                                      'compbinations for', items_to_match)
        result = self.sort_and_store_record(dist_metric, catalogue_name, num_best_maches)
        result['info'].append(wardrobe_obj.data_dict['obj_name'])        
        # result['match_mode'] = match_mode
        return result
        
    

    def user_wardrobe_vs_itself_matches(self, wardrobe_obj, items_to_match = ['jacket', 'pants' ], 
                  num_best_maches = 10, n_split=1, match_mode = 'complement', use_random_split = True,
                  single_query = None):
        '''match_mode: 
            - complement, complement_opposite, complement_shade
        - analogous, analogous_opposite, analogous_shade,     
        num_best_maches: the result may be lower than the selected after removing duplicates 
        n_split: used to split complement and analogous, higher value result in more colors
        n_split=0 only works with complement, it should be >=1 for analogous
        use_random_split: if True, n_split will be generated randomly depending on the ref item, and used 
        to find complement colors splitted on the query 
        single_query = 'jacekt.png' ... will only compare the jacket to the rest of items in items_to_match,
        if None, will compare all to all in items_to_match
        
        '''
        
        ''' IDEA: maybe we can keep only the major color in each item
        then, for generate a complement wheel, randomly pick a color for each of the items '''
        
        dist_metric = []; catalogue_name = []
        self.wardrobe_sanity_check(wardrobe_obj, items_to_match)  
        item_Q, item_R = items_to_match # this mainly has one item, but did a for-loop for generality                                      
                   
        if single_query != None:      
            Q_img_names = [single_query]            
            item_R, item_Q = (item_R, item_Q) if single_query[:single_query.find('_')]==item_Q else (item_Q, item_R)
        else:            
            Q_img_names = wardrobe_obj.data_dict['colr_df'][item_Q].index
        
        R_item_pairs = self.generage_item_pairs(wardrobe_obj, [item_R])    
        for Q_img_name in Q_img_names:          
            self.one_Q_to_many_R_match(Q_img_name, dist_metric, catalogue_name, 
                              n_split, R_item_pairs, item_R, wardrobe_obj, 
                              match_mode, item_Q, use_random_split)                                       
                            
        result = self.sort_and_store_record(dist_metric, catalogue_name, num_best_maches)
        result['info'].append( wardrobe_obj.data_dict['obj_name'])
        result['matching x to y'] =  'Self Wardrobe'
        result['match_mode'] =   match_mode         
        
        return result
    
    
    def one_Q_to_many_R_match(self, Q_img_name, dist_metric, catalogue_name, 
                              n_split, R_item_pairs, item_R, wardrobe_obj, 
                              match_mode, item_Q, use_random_split):        
        # generate the colors of every query_image_name, then, find the distance with each single item
        p_Q = wardrobe_obj.data_dict['pix_cnt_df'][item_Q].loc[Q_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
        c_Q = self.to_2D_mat(wardrobe_obj, item_Q, Q_img_name, 'colr_df')   
        n_split = self.get_split_point(use_random_split, len(self.remove_nans_from_numpy(p_Q)), n_split )
        for R_img_name in R_item_pairs: # for query_img_name in wardrobe_obj.data_dict['colr_df'][item_query].index:                    
            p_R_orig = wardrobe_obj.data_dict['pix_cnt_df'][item_R].loc[R_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
            c_R_orig = self.to_2D_mat(wardrobe_obj, item_R, R_img_name, 'colr_df')
            # now, we have to find the color complement of c_query and compare it with the ref
            R_color_pack = self.get_color_pack(c_R_orig, p_R_orig, 
                                                    n_split, match_mode = match_mode) # we have to take the effect of the original probability and multiply it by the new one from the split, necessary when c_query_orig has more than one RGB with different probabilities
                                    
            for c_R_key in R_color_pack.keys(): # ''' we need a distance for each pack, if it is a multi pack '''
                c_R_instance = R_color_pack[c_R_key]['color']                            
                p_R_instance = R_color_pack[c_R_key]['prob'] # this is the resutlant split probabilit multiplied by the prior p0=p_query_orig, that is, low values p_query_orig should penalize the new colors                                    
                dist_item = self.find_e_distance(c_R_instance, 
                                                  p_R_instance, 
                                                  c_Q, p_Q)  
                dist_metric.append(dist_item) # much easier to store the distance separately, as we need it to find the min values to obtain the best matches
                catalogue_name.append({'matched_items': sorted((R_img_name, Q_img_name))                                        
                                        
                                        }) # registering the catalogue that possibly matches the query
       
    

x = ColorMatching()
catalogue = 'ref_clothCoP'
ref_obj = ColorTable(fashion_obj_name='Catalogue '+catalogue) 
ref_obj.load('ref_clothCoP.pkl')

wardrobe_user_name = 'malrawi'
user_obj = ColorTable(fashion_obj_name = 'Wardrobe of '+ wardrobe_user_name) 
user_obj.load( wardrobe_user_name +'.pkl')

items_to_match = ['shirt',  'pants'] # ['jacket', 'shirt', 'pants']
# result = x.ref_v_ref_matches(ref_obj, user_obj, items_to_match, num_best_maches = 4)
# result = x.user_wardrobe_v_ref_matches(ref_obj, user_obj, 
#                                         items_to_match, 
#                                         num_best_maches = 10)


'''  match_mode
        - complement, complement_opposite, complement_shade
        - analogous, analogous_opposite, analogous_shade,     
'''     

for i in range(100):
    result_1 = x.user_wardrobe_vs_itself_matches(user_obj, 
                                            items_to_match,                                         
                                            num_best_maches = 10, 
                                            n_split = 0,
                                            single_query = 'shirt_1.png',
                                            match_mode = 'analogous_opposite')
    
    
    items_to_match = ['jacket', 'shirt'] 
    result_2 = x.user_wardrobe_vs_itself_matches(user_obj, 
                                            items_to_match,                                         
                                            num_best_maches = 10, 
                                            n_split = 0,
                                            single_query = 'shirt_1.png',
                                            match_mode = 'analogous_opposite')

                
    # def get_colors_from_results(r_obj, result):
    #     ''' TODO '''
    #     for i in range(len(result['catalogue_name'])):
    #         query = result['catalogue_name'][i]['query'] # in the future, this should be a list of images, for each of the matched_items
    #         color_mathces_catalogue_name = result['catalogue_name'][i]['reference']
    #         matched_items = result['catalogue_name'][i]['matched_items']
                            
    #     return 0


# def user_wardrobe_vs_itself_matches_old_nov19_6pm(self, wardrobe_obj, items_to_match = ['jacket', 'pants', 'shirt' ], 
#                   num_best_maches = 10, n_split=1, match_mode = 'complement', use_random_split = False):
#         '''match_mode: complement, analogous, mix (has analogous, complement, opposite, and shade) 
#         num_best_maches: the result may be lower than the selected after removing duplicates 
#         n_split: used to split complement and analogous, higher value result in more colors
#         n_split=0 only works with complement, it should be >=1 for analogous
#         use_random_split: if True, n_split will be generated randomly depending on the ref item, and used 
#         to find complement colors splitted on the query '''
        
#         ''' IDEA: maybe we can keep only the major color in each item
#         then, for generate a complement wheel, randomly pick a color for each of the items '''
        
#         dist_metric = []; catalogue_name = []
#         self.wardrobe_sanity_check(wardrobe_obj, items_to_match)  
#         item_ref = items_to_match[0] # this mainly has one item, but did a for-loop for generality                                      
#         item_query = items_to_match[1]        
#         query_item_pairs = self.generage_item_pairs(wardrobe_obj, [item_query])
        
#         for ref_img_name in wardrobe_obj.data_dict['colr_df'][item_ref].index:
#             p_ref = wardrobe_obj.data_dict['pix_cnt_df'][item_ref].loc[ref_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
#             c_ref = self.to_2D_mat(wardrobe_obj, item_ref, ref_img_name, 'colr_df')   
#             n_split = self.get_split_point(use_random_split, len(self.remove_nans_from_numpy(p_ref)), n_split )
            
#             # generate the colors of every query_image_name, then, find the distance with each single item
#             for query_img_name in query_item_pairs: # for query_img_name in wardrobe_obj.data_dict['colr_df'][item_query].index:                    
#                 p_query_orig = wardrobe_obj.data_dict['pix_cnt_df'][item_query].loc[query_img_name].values # for the wardrobe, the image_name is used instead of the item in the pandas table
#                 c_query_orig = self.to_2D_mat(wardrobe_obj, item_query, query_img_name, 'colr_df')
#                 # now, we have to find the color complement of c_query and compare it with the ref
#                 query_color_pack = self.get_color_pack(c_query_orig, p_query_orig, 
#                                                         n_split, match_mode = match_mode) # we have to take the effect of the original probability and multiply it by the new one from the split, necessary when c_query_orig has more than one RGB with different probabilities
                                        
#                 for c_query_key in query_color_pack.keys(): # ''' we need a distance for each pack, if it is a multi pack '''
#                     c_query_instance = query_color_pack[c_query_key]['color']                            
#                     p_query_instance = query_color_pack[c_query_key]['prob'] # this is the resutlant split probabilit multiplied by the prior p0=p_query_orig, that is, low values p_query_orig should penalize the new colors                                    
#                     dist_item = self.find_e_distance(c_query_instance, 
#                                                       p_query_instance, 
#                                                       c_ref, p_ref)  
#                     dist_metric.append(dist_item) # much easier to store the distance separately, as we need it to find the min values to obtain the best matches
#                     catalogue_name.append({'query': items_to_match, 
#                                             'reference': 'Self Wardrobe', 
#                                             'matched_items': sorted((query_img_name, ref_img_name)),
#                                             'color-match': match_mode
#                                             }) # registering the catalogue that possibly matches the query
                                             
                        
#         result = self.sort_and_store_record(dist_metric, catalogue_name, num_best_maches)
#         result['info'].append(wardrobe_obj.data_dict['obj_name'])        
#         return result
    
    