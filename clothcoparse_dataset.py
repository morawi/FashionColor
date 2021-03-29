import glob
import os
import scipy.io as sio
from torch.utils.data import Dataset # Dataset class from PyTorch
from PIL import Image, ImageChops # PIL is a nice Python Image Library that we can use to handle images
import numpy as np


# https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
        

class ImageDataset(Dataset):
    def __init__(self, root, class_names_and_colors, mode="train",  HPC_run=False):
        
        self.class_names = list(class_names_and_colors.keys())
        
        if HPC_run:
            root = '/home/malrawi/MyPrograms/Data/ClothCoParse'
        
        self.files_A = sorted(glob.glob(os.path.join(root, "%s/A" % mode) + "/*.*")) # get the source image file-names
        self.files_B = sorted(glob.glob(os.path.join(root, "%s/B" % mode) + "/*.*")) # get the target image file-names
        
    def number_of_classes(self, opt):
        return(len(self.class_names)) # this should do
  

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
            labels.append(self.class_names[obj_ids[i]])
                              
        image_id = index
        fname = os.path.basename(self.files_A[index % len(self.files_A)])          
        
        return image_A, masked_img, labels, image_id, masks, fname
     

    def __len__(self): # this function returns the length of the dataset, the source might not equal the target if the data is unaligned
        return len(self.files_B)

