# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:47:11 2021

@author: 
    https://github.com/miranov25/RootInteractive/blob/b54446e09072e90e17f3da72d5244a20c8fdd209/RootInteractive/Tools/Histograms/histogramdd.py
"""

"""
hsgh
"""

import torch

_range = range

def histogramdd(sample,bins=None,range_=None,weights=None,remove_overflow=True):
    edges=None
    device=None
    custom_edges = False
    D,N = sample.shape
    if device == None:
        device = sample.device
    if bins == None:
        if edges == None:
            bins = 10
            custom_edges = False
        else:
            try:
                bins = edges.size(1)-1
            except AttributeError:
                bins = torch.empty(D)
                for i in _range(len(edges)):
                    bins[i] = edges[i].size(0)-1
                bins = bins.to(device)
            custom_edges = True
    try:
        M = bins.size(0)
        if M != D:
            raise ValueError(
                'The dimension of bins must be equal to the dimension of the '
                ' sample x.')
    except AttributeError:
        # bins is either an integer or a list
        if type(bins) == int:
            bins = torch.full([D],bins,dtype=torch.long,device=device)
        elif torch.is_tensor(bins[0]):
            custom_edges = True
            edges = bins
            bins = torch.empty(D,dtype=torch.long)
            for i in _range(len(edges)):
                bins[i] = edges[i].size(0)-1
            bins = bins.to(device)
        else:
            bins = torch.as_tensor(bins)
    if bins.dim() == 2:
        custom_edges = True
        edges = bins
        bins = torch.full([D],bins.size(1)-1,dtype=torch.long,device=device)
    if custom_edges:
        use_old_edges = False
        if not torch.is_tensor(edges):
            use_old_edges = True
            edges_old = edges
            m = max(i.size(0) for i in edges)
            tmp = torch.empty([D,m],device=edges[0].device)
            for i in _range(D):
                s = edges[i].size(0)
                tmp[i,:]=edges[i][-1]
                tmp[i,:s]=edges[i][:]
            edges = tmp.to(device)
        k = torch.searchsorted(edges,sample)
        k = torch.min(k,(bins+1).reshape(-1,1))
        if use_old_edges:
            edges = edges_old
        else:
            edges = torch.unbind(edges)
    else:
            if range_ == None: #range is not defined
                range_ = torch.empty(2,D,device=device)
                if N == 0: #Empty histogram
                    range_[0,:] = 0
                    range_[1,:] = 1
                else:
                    range_[0,:]=torch.min(sample,1)[0]
                    range_[1,:]=torch.max(sample,1)[0]
            elif not torch.is_tensor(range_): #range is a tuple
                r = torch.empty(2,D)
                for i in _range(D):
                    if range_[i] is not None:
                        r[:,i] = torch.as_tensor(range_[i])
                    else:
                        if N == 0: #Edge case: empty histogram
                            r[0,i] = 0
                            r[1,i] = 1
                        r[0,i]=torch.min(sample[:,i])[0]
                        r[1,i]=torch.max(sample[:,i])[0]
                range_ = r.to(device=device,dtype=sample.dtype)
            singular_range = torch.eq(range_[0],range_[1]) #If the range consists of only one point, pad it up
            range_[0,singular_range] -= .5
            range_[1,singular_range] += .5
            
            xx = range_.int().cpu().numpy()
            edges = [torch.linspace(xx[0,i], xx[1,i], bins[i].int().cpu()+1) for i in _range(len(bins))]
            tranges = torch.empty_like(range_)
            tranges[1,:] = bins/(range_[1,:]-range_[0,:])
            tranges[0,:] = 1-range_[0,:]*tranges[1,:]
            k = torch.addcmul(tranges[0,:].reshape(-1,1),sample,tranges[1,:].reshape(-1,1)).long() #Get the right index
            k = torch.max(k,torch.zeros([],device=device,dtype=torch.long)) #Underflow bin
            k = torch.min(k,(bins+1).int().reshape(-1,1))


    multiindex = torch.ones_like(bins)
    multiindex[1:] = torch.cumprod(torch.flip(bins[1:],[0])+2,-1).long()
    multiindex = torch.flip(multiindex,[0])
    l = torch.sum(k*multiindex.reshape(-1,1),0)
    hist = torch.bincount(l,minlength=(multiindex[0]*(bins[0]+2)).item(),weights=weights)
    hist = hist.reshape(tuple(bins+2))
    if remove_overflow:
        core = D * (slice(1, -1),)
        hist = hist[core]
    return hist,edges