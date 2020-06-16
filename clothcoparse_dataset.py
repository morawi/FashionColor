import glob
import os
import scipy.io as sio
from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image, ImageChops # PIL is a nice Python Image Library that we can use to handle images
import torchvision.transforms as transforms # torch transform used for computer vision applications
import numpy as np
import torch
# import sys

# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html


def get_clothCoParse_class_names(): 
    # names ordered according to label id, 0 for background and 59 for wedges
    
    ClothCoParse_class_names = ['background',  'accessories',  'bag',  'belt',  'blazer',
 'blouse',  'bodysuit',  'boots',  'bra',  'bracelet',  'cape',  'cardigan',
 'clogs', 'coat',  'dress', 'earrings', 'flats', 'glasses', 'gloves', 'hair',
 'hat', 'heels', 'hoodie', 'intimate', 'jacket', 'jeans', 'jumper', 'leggings',
 'loafers', 'necklace', 'panties', 'pants', 'pumps', 'purse', 'ring', 'romper',
 'sandals', 'scarf', 'shirt', 'shoes', 'shorts', 'skin', 'skirt', 'sneakers',
 'socks', 'stockings', 'suit', 'sunglasses', 'sweater', 'sweatshirt', 'swimwear',
 't-shirt', 'tie', 'tights', 'top', 'vest', 'wallet', 'watch', 'wedges']
    
    return ClothCoParse_class_names
        

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, transforms_target=None,
                 mode="train", person_detection=False,
                 HPC_run=False, remove_background=True,
                 ):
       
        if transforms_ != None:
            self.transforms = transforms.Compose(transforms_) # image transform
        else: self.transforms=None
        if transforms_target != None:
            self.transforms_target = transforms.Compose(transforms_target) # image transform
        else: self.transforms_target=None
        
        self.class_name = get_clothCoParse_class_names()
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/ClothCoParse'
        
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*")) # get the source image file-names
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*")) # get the target image file-names
        
    def number_of_classes(self, opt):
        if opt.person_detection:
            return 2
        else:
            return(len(get_clothCoParse_class_names())) # this should do
  

    def __getitem__(self, index):              
                
        annot = sio.loadmat(self.files_B[index % len(self.files_B)])
        mask = annot["groundtruth"]
        image_A = Image.open(self.files_A[index % len(self.files_A)]) # read the image, according to the file name, index select which image to read; index=1 means get the first image in the list self.files_A
       
        # instances are encoded as different colors
        obj_ids = np.unique(mask)[1:] # first id is the background, so remove it     
        masks = mask == obj_ids[:, None, None] # split the color-encoded mask into a set of binary masks

        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)                       
        masked_img = []; labels =[]
        for i in range(num_objs):
            img = ImageChops.multiply(image_A, Image.fromarray(255*masks[i]).convert('RGB') )
            masked_img.append(np.array(img, dtype='uint8'))                               
            labels.append(self.class_name[obj_ids[i]])
                              
        image_id = index
        fname = self.files_A[index % len(self.files_A)][-8:-4]
          
        
        return image_A, masked_img, labels, image_id, masks, fname
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files_B)

