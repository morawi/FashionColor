# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 15:25:59 2021

@author: malrawi
"""

import torchvision
import torch
import numpy as np
from colors_from_hist_torch import colors_via_3d_hist
import time
# import kornia


'''

torchvision.datasets.ImageFolder(root: str, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, loader: Callable[[str], Any] = <function default_loader>, is_valid_file: Union[Callable[[str], bool], NoneType] = None)[SOURCE]
A generic data loader where the images are arranged in this way:

root/dog/xxx.png
root/dog/xxy.png
root/dog/[...]/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/[...]/asd932_.png

'''



# Class to perform the padding
class RemoveBackground(object):
    """
            Resizing will be skipped if max_allowed_size is None
            
    """
    def __init__(self, max_allowed_size = 160000): 
        ''' Resizing will be skipped if max_allowed_size is None
             '''
        self.max_allowed_size = max_allowed_size 
            

    def __call__(self, image):
        img_shape = image.shape
        img_size = img_shape[1]*img_shape[2]
        
        if self.max_allowed_size is not None and img_size>self.max_allowed_size :   # scale the image down, keep aspect ratio, if it larger than max_allowed_size         
            scale_factor = np.sqrt(self.max_allowed_size/img_size)        
            new_shape = [int(scale_factor*img_shape[1]), int(scale_factor*img_shape[2])]
            image = torchvision.transforms.Resize(new_shape, 
                      interpolation= torchvision.transforms.InterpolationMode.NEAREST)(image)
        
        mask = image[3, :, :].bool() # this is the alpha channel ... 
                
        _size = mask.shape[0]*mask.shape[1]
        mask = mask.view(_size)
        image = image[0:3,:,:] # get the RGB image stored as CxWxH
        # image = image.permute(1, 2, 0) # this might not be needed, we permuted it below        
        image = torch.stack(( image[1].view(-1), image[2].view(-1), image[0].view(-1)), -1) # convert to an array of 3D values
        
        
        image = image[mask]       # remove background      
        
        return image


class ImageFolder(torchvision.datasets.ImageFolder):
    def __init__(self, root, 
                 transform=torchvision.transforms.Compose([RemoveBackground(max_allowed_size = 160000)]), 
                 target_transform=None, loader=torchvision.io.read_image):
       super().__init__(root, transform, target_transform, loader)    
                                 
    def __len__(self):
        return len(self.samples)       
        
    def __getitem__(self, index):        
        fname = self.imgs[index][0].strip(self.root)        
        return super(ImageFolder, self).__getitem__(index)[0], fname        
    
          
tic =  time.time()                            


max_allowed_size = 160000
xx = ImageFolder(root = '../Data/Wardrobe/', 
                  transform = torchvision.transforms.Compose([RemoveBackground(max_allowed_size = max_allowed_size)]), 
                  loader = torchvision.io.read_image)

# zz = torch.rand(2, 3, 4, 5)
# output = kornia.rgb_to_hsv(zz)  # 2x3x4x5

# xx[0][0]
for image_no_bkg in xx:        
    print('\n \n ----------------')
    print(image_no_bkg[1])
    peaked_colors = colors_via_3d_hist(image_no_bkg[0], verbose=True)    

print('\n \n elapsed time', time.time()-tic)


# import math
# import numbers
# import torch
# from torch import nn
# from torch.nn import functional as F

# class GaussianSmoothing(nn.Module):
#     """
#     Apply gaussian smoothing on a
#     1d, 2d or 3d tensor. Filtering is performed seperately for each channel
#     in the input using a depthwise convolution.
#     Arguments:
#         channels (int, sequence): Number of channels of the input tensors. Output will
#             have this number of channels as well.
#         kernel_size (int, sequence): Size of the gaussian kernel.
#         sigma (float, sequence): Standard deviation of the gaussian kernel.
#         dim (int, optional): The number of dimensions of the data.
#             Default value is 2 (spatial).
#     """
#     def __init__(self, channels, kernel_size, sigma, dim=2):
#         super(GaussianSmoothing, self).__init__()
#         if isinstance(kernel_size, numbers.Number):
#             kernel_size = [kernel_size] * dim
#         if isinstance(sigma, numbers.Number):
#             sigma = [sigma] * dim

#         # The gaussian kernel is the product of the
#         # gaussian function of each dimension.
#         kernel = 1
#         meshgrids = torch.meshgrid(
#             [
#                 torch.arange(size, dtype=torch.float32)
#                 for size in kernel_size
#             ]
#         )
#         for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
#             mean = (size - 1) / 2
#             kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
#                       torch.exp(-((mgrid - mean) / std) ** 2 / 2)

#         # Make sure sum of values in gaussian kernel equals 1.
#         kernel = kernel / torch.sum(kernel)

#         # Reshape to depthwise convolutional weight
#         kernel = kernel.view(1, 1, *kernel.size())
#         kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

#         self.register_buffer('weight', kernel)
#         self.groups = channels

#         if dim == 1:
#             self.conv = F.conv1d
#         elif dim == 2:
#             self.conv = F.conv2d
#         elif dim == 3:
#             self.conv = F.conv3d
#         else:
#             raise RuntimeError(
#                 'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
#             )

#     def forward(self, input):
#         """
#         Apply gaussian filter to input.
#         Arguments:
#             input (torch.Tensor): Input to apply gaussian filter on.
#         Returns:
#             filtered (torch.Tensor): Filtered output.
#         """
#         return self.conv(input, weight=self.weight, groups=self.groups)


# smoothing = GaussianSmoothing(3, 5, 1)
# input = torch.rand(1, 3, 100, 100)
# input = F.pad(input, (2, 2, 2, 2), mode='reflect')
# output = smoothing(input)




# zz = torchvision.datasets.ImageFolder(root=path2folder,
#                                       transform = remBkg_trans,
#                                       loader = torchvision.io.read_image) #,
