#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 12:39:36 2021

@author: jtick20
"""


import numpy as np
 
import torch
from torch import nn
from torch import optim
import torch.fft
import matplotlib.pyplot as plt
import tensorboardX

import h5py
import os 
from os.path import exists



from complexLayers3D import NaiveComplexBatchNorm3d, ComplexConv3d, ComplexMaxPool3d, ComplexConvTranspose3d,  ComplexReLU

from complexFunctions3D import complex_upsample


import math

import time

##############################################################################
# Interpolation

from itertools import product

assert hasattr(
    torch, "bucketize"), "Need torch >= 1.7.0; install at pytorch.org"


class RegularGridInterpolator:

    def __init__(self, points, values):
        self.points = points
        self.values = values

        assert isinstance(self.points, tuple) or isinstance(self.points, list)
        assert isinstance(self.values, torch.Tensor)

        self.ms = list(self.values.shape)
        self.n = len(self.points)

        assert len(self.ms) == self.n

        for i, p in enumerate(self.points):
            assert isinstance(p, torch.Tensor)
            assert p.shape[0] == self.values.shape[i]

    def __call__(self, points_to_interp):
        assert self.points is not None
        assert self.values is not None

        assert len(points_to_interp) == len(self.points)
        K = points_to_interp[0].shape[0]
        for x in points_to_interp:
            assert x.shape[0] == K

        idxs = []
        dists = []
        overalls = []
        for p, x in zip(self.points, points_to_interp):
            idx_right = torch.bucketize(x, p)
            idx_right[idx_right >= p.shape[0]] = p.shape[0] - 1
            idx_left = (idx_right - 1).clamp(0, p.shape[0] - 1)
            dist_left = x - p[idx_left]
            dist_right = p[idx_right] - x
            dist_left[dist_left < 0] = 0.
            dist_right[dist_right < 0] = 0.
            both_zero = (dist_left == 0) & (dist_right == 0)
            dist_left[both_zero] = dist_right[both_zero] = 1.

            idxs.append((idx_left, idx_right))
            dists.append((dist_left, dist_right))
            overalls.append(dist_left + dist_right)

        numerator = 0.
        for indexer in product([0, 1], repeat=self.n):
            as_s = [idx[onoff] for onoff, idx in zip(indexer, idxs)]
            bs_s = [dist[1 - onoff] for onoff, dist in zip(indexer, dists)]
            numerator += self.values[as_s] * \
                torch.prod(torch.stack(bs_s), dim=0)
        denominator = torch.prod(torch.stack(overalls), dim=0)
        return numerator / denominator





##############################################################################

# Data loading

def extract_images(filename,imageName):
  """Extract the images into a 4D uint8 numpy array."""
  fData = h5py.File(filename,'r')
  inData = fData.get(imageName)  
      
  
  num_images = inData.shape[0]
  rows = inData.shape[1]
  cols = inData.shape[2]
  zcols = inData.shape[3]
  
  print('Data size of: ' + imageName)
  print(num_images, rows, cols,zcols)
  data = np.array(inData)
    
  data = np.array(data.tolist())
     
    
  data = data.reshape(num_images,1, rows, cols, zcols)
  
  return data


class DataSet(object):

  def __init__(self, data, true, ind):
    """Construct a DataSet"""

    assert data.shape[0] == true.shape[0], (
        'images.shape: %s labels.shape: %s' % (data.shape,
                                                 true.shape))
    data = data.reshape(data.shape[0],
                            data.shape[1],data.shape[2],data.shape[3], data.shape[4] )
    true = true.reshape(true.shape[0],
                            true.shape[1],true.shape[2],true.shape[3],true.shape[4])
    
    self._num_examples = data.shape[0]

    self._data = data
    self._true = true
    
    self._data_orig = data
    self._true_orig = true
    
    self._epochs_completed = 0
    self._index_in_epoch = 0
    
    self._ind = ind

  @property
  def data(self):
    return self._data

  @property
  def true(self):
    return self._true
 


  @property
  def data_orig(self):
    return self._data_orig

  @property
  def true_orig(self):
    return self._true_orig


  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_completed(self):
    return self._epochs_completed

  @property
  def ind(self):
    return self._ind 

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._data = self._data[perm]
      self._true = self._true[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._data[start:end], self._true[start:end]

  def selected_set(self, ind):

      
      return self._data_orig[ind], self._true_orig[ind]



def read_data_sets(FileNameTrain,FileNameTest, indTrain, indTest):
  class DataSets(object):
    pass
  data_sets = DataSets()

  TRAIN_SET = FileNameTrain
  TEST_SET  = FileNameTest
  TRUE_NAME  = 'imagesTrue'
  DATA_NAME  = 'dataTrue'
  
  print('Start loading data') 
  print('Training data') 
  train_true   = extract_images(TRAIN_SET,TRUE_NAME)
  train_data   = extract_images(TRAIN_SET,DATA_NAME)
   
  print('Testing data') 
  test_true   = extract_images(TEST_SET,TRUE_NAME)
  test_data   = extract_images(TEST_SET,DATA_NAME)
  


  data_sets.train = DataSet(train_data, train_true, indTrain)
  data_sets.test = DataSet(test_data, test_true, indTest)

  return data_sets


##############################################################################
# Network functions 

def complex_relu2(input, inplace):
    return torch.nn.functional.relu(input.real, inplace).type(torch.complex64)+1j*torch.nn.functional.relu(input.imag, inplace).type(torch.complex64)

class ComplexUpsample(nn.Module):

    def __init__(self, size=None, scale_factor=None, mode='nearest',
                             align_corners=None, recompute_scale_factor=None):
        super(ComplexUpsample,self).__init__()
        self.size=size
        self.scale_factor=scale_factor
        self.mode=mode
        self.align_corners=align_corners
        self.recompute_scale_factor=recompute_scale_factor


    def forward(self,input):
        return complex_upsample(input, size=self.size, scale_factor=self.scale_factor, mode=self.mode,
                             align_corners=self.align_corners, recompute_scale_factor=self.recompute_scale_factor)


class ComplexReLU2(nn.Module):
    
    def __init__(self, inplace=True):
        super(ComplexReLU2,self).__init__()
        self.inplace=inplace

    def forward(self,input):
        return complex_relu2(input, inplace=self.inplace)




def complex_double_conv(in_channels, out_channels):
    return nn.Sequential(
       ComplexConv3d(in_channels, out_channels, 3, padding=1),
       NaiveComplexBatchNorm3d(out_channels),       
       ComplexReLU(),
       ComplexConv3d(out_channels, out_channels, 3, padding=1),
       NaiveComplexBatchNorm3d(out_channels),
       ComplexReLU())  

   
def double_conv(in_channels, out_channels):
    return nn.Sequential(
       nn.Conv3d(in_channels, out_channels, 3, padding=1),
       nn.BatchNorm3d(out_channels),       
       nn.ReLU(inplace=True),
       nn.Conv3d(out_channels, out_channels, 3, padding=1),
       nn.BatchNorm3d(out_channels),
       nn.ReLU(inplace=True))

##############################################################################
# Inverse operators    
   
class PATinv(nn.Module):
    def __init__(self, sf, points_inv, pointsI_inv , c , NtFactor, NxyzI):
        super().__init__()
        self.sf=sf
        self.points_inv=points_inv
        self.pointsI_inv=pointsI_inv
        self.c=c
        self.NtFactor=NtFactor
        self.NxyzI=NxyzI
        

    def forward(self, pT):
        
        
        pUp= torch.flipud(pT[0,0,:,:,:])
        p0 = torch.cat((pUp[None,None,:,:, :],pT[:,:,1::,:, :]),2)
        
        indS=math.ceil((p0.shape[2]+1)/2/self.NtFactor)
        indE=math.floor((p0.shape[2])/self.NtFactor)+1
        
        
        p0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(p0)))
        p0 = p0*self.sf
    

        
        p0=p0[0,0]
        
       
        
        gi = RegularGridInterpolator(self.points_inv, p0)
        
        p0=gi(self.pointsI_inv)
        
     
        
        p0=torch.reshape(p0, (1, 1, self.NxyzI[0], self.NxyzI[1], self.NxyzI[2] ))
        
      

        p0 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(p0)))
         
        p0=2*2*p0.real[:,:,indS:indE, :, :]/self.c /self.NtFactor
        
        
        
        return p0    
    



   
class PATinvNN(nn.Module):
    def __init__(self,sf, points_inv, pointsI_inv, c, NtFactor, NxyzI, Unet):
        super().__init__()
        self.sf=sf
        self.points_inv=points_inv
        self.pointsI_inv=pointsI_inv
        self.c=c
        self.NtFactor=NtFactor
        self.Unet=Unet
        self.NxyzI=NxyzI

    def forward(self, pT):
        
    
        
        
        pUp= torch.flipud(pT[0,0,:,:,:])
        p0 = torch.cat((pUp[None,None,:,:, :],pT[:,:,1::,:, :]),2)
        
        
        
        indS=math.ceil((p0.shape[2]+1)/2/self.NtFactor)
        indE=math.floor((p0.shape[2])/self.NtFactor)+1
        
        
        
        p0 = torch.fft.fftshift(torch.fft.fftn(torch.fft.ifftshift(p0)))
        p0 = p0*self.sf
        
        
    
                
        p0=p0[0,0]
        
       
        
        gi = RegularGridInterpolator(self.points_inv, p0)
        
        p0=gi(self.pointsI_inv)
        
     
        
        p0=torch.reshape(p0, (1, 1, self.NxyzI[0], self.NxyzI[1], self.NxyzI[2] ))
        
      
        
        
        
        p0I= self.Unet(p0)

        p0 = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(p0)))
        
        p0I = torch.fft.fftshift(torch.fft.ifftn(torch.fft.ifftshift(p0I)))
         
        
        p0=2*2*p0.real[:,:,indS:indE, :, :]/self.c /self.NtFactor
        p0I=2*2*p0I.real[:,:,indS:indE, :, :]/self.c /self.NtFactor
        
     
      
    
        return p0, p0I

##############################################################################
# Networks

class UNetComplex(nn.Module):
    def __init__(self, n_in, n_out, width_channels, device):
        super().__init__()
               
        self.dconv_data1 = complex_double_conv(n_in, width_channels)
        self.dconv_data2 = complex_double_conv(width_channels, 1 )
        
        self.dconv_down1 = complex_double_conv(n_in, width_channels)
        self.dconv_down2 = complex_double_conv(width_channels, width_channels*2)
        self.dconv_down3 = complex_double_conv(width_channels*2, width_channels*4)

       
        self.maxpool = ComplexMaxPool3d(2,ceil_mode=True)
        self.upsample = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = ComplexConvTranspose3d(width_channels*4,width_channels*2,2,stride=2,padding=0)
        self.xUp1  = ComplexConvTranspose3d(width_channels*2,width_channels,2,stride=2,padding=0)
        

        self.dconv_up2 = complex_double_conv(width_channels*2 + width_channels*2, width_channels*2)
        self.dconv_up1 = complex_double_conv(width_channels + width_channels, width_channels)
        self.conv_last = ComplexConv3d(width_channels, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        
        self.device = device

    def forward(self, p):
    
        
        
        conv1 = self.dconv_down1(p)
        x = self.maxpool(conv1)
       
        
    
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        
        
        conv3 = self.dconv_down3(x)
        
        
        x = self.xUp2(conv3) 
        
               
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(x)
    
       
        
        x = self.xUp1(x) 
        
        
    
        
        x = torch.cat([x, conv1], dim=1) 
        x = self.dconv_up1(x)
       
        
        
        
        
        update = self.conv_last(x)
      
        
        

        return self.stepsize * update



class UNetComplex2(nn.Module):
    def __init__(self, n_in, n_out, width_channels, device):
        super().__init__()
               
        self.dconv_data1 = complex_double_conv(n_in, width_channels)
        self.dconv_data2 = complex_double_conv(width_channels, 1 )
        
        self.dconv_down1 = complex_double_conv(n_in, width_channels)
        self.dconv_down2 = complex_double_conv(width_channels, width_channels*2)
        self.dconv_down3 = complex_double_conv(width_channels*2, width_channels*4)
        self.dconv_down4 = complex_double_conv(width_channels*4, width_channels*8)

       
        self.maxpool = ComplexMaxPool3d(2,ceil_mode=True)
        self.upsample = ComplexUpsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp3  = ComplexConvTranspose3d(width_channels*8,width_channels*4,2,stride=2,padding=0)
        self.xUp2  = ComplexConvTranspose3d(width_channels*4,width_channels*2,2,stride=2,padding=0)
        self.xUp1  = ComplexConvTranspose3d(width_channels*2,width_channels,2,stride=2,padding=0)
        

        self.dconv_up3 = complex_double_conv(width_channels*4 + width_channels*4, width_channels*4)
        self.dconv_up2 = complex_double_conv(width_channels*2 + width_channels*2, width_channels*2)
        self.dconv_up1 = complex_double_conv(width_channels + width_channels, width_channels)
        self.conv_last = ComplexConv3d(width_channels, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        
        self.device = device

    def forward(self, p):
    
        
        
        conv1 = self.dconv_down1(p)
        x = self.maxpool(conv1)
       
        
    
        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
        
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        
        conv4 = self.dconv_down4(x)
        
        
        x = self.xUp3(conv4) 
        
        x = torch.cat([x, conv3], dim=1) 
        x = self.dconv_up3(x)
        
        
        
        x = self.xUp2(x) 
        
               
        x = torch.cat([x, conv2], dim=1) 
        x = self.dconv_up2(x)
    
       
        
        x = self.xUp1(x) 
        
        
    
        
        x = torch.cat([x, conv1], dim=1) 
        x = self.dconv_up1(x)
       
        
      
        
        update = self.conv_last(x)
      
        
        

        return self.stepsize * update





class resNetComplex(nn.Module):
    def __init__(self,n_in,n_out, width_channels):
        super().__init__()
        self.doubleConv = nn.Sequential(
           ComplexConv3d(n_in, width_channels, 3, padding=1),
           NaiveComplexBatchNorm3d(width_channels),   
           ComplexReLU(),
           ComplexConv3d(width_channels, width_channels, 3, padding=1),
           NaiveComplexBatchNorm3d(width_channels),   
           ComplexReLU(),
           ComplexConv3d(width_channels, n_out, 3, padding=1,bias=False)
       )
        
    def forward(self, cur):
        
        update = self.doubleConv(cur)
        
        return update


 
class resNetPP(nn.Module):
    def __init__(self,n_in,n_out, width_channels):
        super().__init__()
        self.doubleConv = nn.Sequential(
           nn.Conv3d(n_in, width_channels, 3, padding=1),
           nn.BatchNorm3d(width_channels),   
           nn.ReLU(inplace=True),
           nn.Conv3d(width_channels, width_channels, 3, padding=1),
           nn.BatchNorm3d(width_channels),   
           nn.ReLU(inplace=True),
           nn.Conv3d(width_channels, n_out, 3, padding=1,bias=False)
       )
        
    def forward(self, cur):
        
        update = self.doubleConv(cur)
        
        return cur+update     
        


    

class UNetPP(nn.Module):
    def __init__(self, n_in, n_out, width_channels):
        super().__init__()
               
        
        self.dconv_down1 = double_conv(n_in, width_channels)
        self.dconv_down2 = double_conv(width_channels, width_channels*2)
        self.dconv_down3 = double_conv(width_channels*2, width_channels*4)

       
        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp2  = nn.ConvTranspose3d(width_channels*4,width_channels*2,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose3d(width_channels*2,width_channels,2,stride=2,padding=0)
        

        self.dconv_up2 = double_conv(width_channels*2 + width_channels*2,width_channels*2)
        self.dconv_up1 = double_conv(width_channels + width_channels, width_channels)
        self.conv_last = nn.Conv3d(width_channels, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        

    def forward(self, inp):
        
      
        
        conv1 = self.dconv_down1(inp)
        x = self.maxpool(conv1)
        
       
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
       
        conv3 = self.dconv_down3(x)
        
        
        x = self.xUp2(conv3)  

              
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        x = self.xUp1(x)        
        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp+self.stepsize * update


class UNetPP2(nn.Module):
    def __init__(self, n_in, n_out, width_channels):
        super().__init__()
               
        
        self.dconv_down1 = double_conv(n_in, width_channels)
        self.dconv_down2 = double_conv(width_channels, width_channels*2)
        self.dconv_down3 = double_conv(width_channels*2, width_channels*4)
        self.dconv_down4 = double_conv(width_channels*4, width_channels*8)

       
        self.maxpool = nn.MaxPool3d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        self.xUp3  = nn.ConvTranspose3d(width_channels*8,width_channels*4,2,stride=2,padding=0)
        self.xUp2  = nn.ConvTranspose3d(width_channels*4,width_channels*2,2,stride=2,padding=0)
        self.xUp1  = nn.ConvTranspose3d(width_channels*2,width_channels,2,stride=2,padding=0)
        

        self.dconv_up3 = double_conv(width_channels*4 + width_channels*4,width_channels*4)
        self.dconv_up2 = double_conv(width_channels*2 + width_channels*2,width_channels*2)
        self.dconv_up1 = double_conv(width_channels + width_channels, width_channels)
        self.conv_last = nn.Conv3d(width_channels, n_out, 1)
        
        self.stepsize = nn.Parameter(torch.zeros(1, 1, 1, 1, 1))
        

    def forward(self, inp):
        
      
        
        conv1 = self.dconv_down1(inp)
        x = self.maxpool(conv1)
        

        
        conv2 = self.dconv_down2(x)
        x = self.maxpool(conv2)
        
         
        conv3 = self.dconv_down3(x)
        x = self.maxpool(conv3)
        
        conv4 = self.dconv_down4(x)
        
        x = self.xUp3(conv4)  

             
        x = torch.cat([x, conv3], dim=1)      
        x = self.dconv_up3(x)
        
        
        x = self.xUp2(x)  
        
        x = torch.cat([x, conv2], dim=1)      
        x = self.dconv_up2(x)
        
        
        
        x = self.xUp1(x)        
        x = torch.cat([x, conv1], dim=1)         
        x = self.dconv_up1(x)
        update = self.conv_last(x)

        return inp+self.stepsize * update


    
 


##############################################################################
# Reconstruction modules
  
class postProcessNetwork(nn.Module):
    def __init__(self, op_adj, UNet):
        super().__init__()
        self.op_adj =op_adj
        self.UNet = UNet
        
   


    def forward(self, pT):
        
       
        p0 = self.op_adj(pT)
         
        
        p0u=self.UNet(p0)
            
        return p0, p0u  
    
    
class postProcessNetworkinvNN(nn.Module):
    def __init__(self, op_adj, UNet):
        super().__init__()
        self.op_adj =op_adj
        self.UNet = UNet
        
   


    def forward(self, pT):
        
       
        p0, p0I = self.op_adj(pT)
         
        
        p0u=self.UNet(p0I)
            
        return p0, p0I, p0u         



##############################################################################
# Summary writers

def summary_image_impl(writer, name, tensor, it):
    image=torch.max(tensor[0,0], 0).values
    image=(image-torch.min(image)) / (torch.max(image)-torch.min(image))
    writer.add_image(name, image, it, dataformats='HW')



def summary_image(writer, name, tensor, it, window=False):
    summary_image_impl(writer, name + '/full', tensor, it)
    if window:
        summary_image_impl(writer, name + '/window', (tensor), it)        
        
        

def summaries(writer, result, fbp, true, resultpT, fbppT, truepT,  loss, it, do_print=False):
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true) - torch.min(true)
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    
    
    relative = torch.mean((result - true) ** 2) / torch.mean((fbp - true) ** 2)


    residualpT = resultpT - truepT
    squared_errorpT = residualpT ** 2
    msepT = torch.mean(squared_errorpT)
    maxvalpT = torch.max(truepT) - torch.min(truepT)
    psnrpT = 20 * torch.log10(maxvalpT) - 10 * torch.log10(msepT)
    
    
    relativepT = torch.mean((resultpT - truepT) ** 2) / torch.mean((fbppT - truepT) ** 2)
    
    if do_print:
            print(it,'MSE p0 ', mse.item(), 'PSNR p0 ', psnr.item(), 'RELATIVE p0 ',  relative.item(),
                  '\n MSE pT ', msepT.item(), 'PSNR pT ', psnrpT.item(), 'RELATIVE pT ',  relativepT.item())


    writer.add_scalar('loss', loss, it)
    
    writer.add_scalar('mse p0', mse, it)
    writer.add_scalar('psnr p0', psnr, it)
    writer.add_scalar('relative p0', relative, it)
    
    writer.add_scalar('mse pT', msepT, it)
    writer.add_scalar('psnr pT', psnrpT, it)
    writer.add_scalar('relative pT', relativepT, it)

    summary_image(writer, 'result p0', result, it)
    summary_image(writer, 'true p0', true, it)
    summary_image(writer, 'fbp p0', fbp, it)
    
    summary_image(writer, 'result pT', resultpT, it)
    summary_image(writer, 'true pT', truepT, it)
    summary_image(writer, 'fbp pT', fbppT, it)
    
    
    
def summaries1(writer, result, fbp, true, loss, it, do_print=False):
    residual = result - true
    squared_error = residual ** 2
    mse = torch.mean(squared_error)
    maxval = torch.max(true) - torch.min(true)
    psnr = 20 * torch.log10(maxval) - 10 * torch.log10(mse)
    
    
    relative = torch.mean((result - true) ** 2) / torch.mean((fbp - true) ** 2)


  
    if do_print:
            print(it,'MSE ', mse.item(), 'PSNR ', psnr.item(), 'RELATIVE ',  relative.item())
                  

    writer.add_scalar('loss', loss, it)
    
    writer.add_scalar('mse', mse, it)
    writer.add_scalar('psnr', psnr, it)
    writer.add_scalar('relative', relative, it)
    

    summary_image(writer, 'result', result, it)
    summary_image(writer, 'true', true, it)
    summary_image(writer, 'fbp', fbp, it)
    

###############################################################################    
#Check parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)   


##############################################################################
# Reconstruction functions

def postProcess(dataSet,
             sfi,ky, w, kz, kyI, wI, kzI,ci, 
             NtFactor,
             expName,filePath,
             netType = 'resNet', 
             lossFunc = 'l2_loss_1',
             bSize = 1,
             trainIter = 5001,
             useTensorboard = True,
             lValInit=1e-3): 
    
              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
  
        
    print('Computations are done using: ' + str(device))
        
    if(useTensorboard):
        
        train_writer_main = tensorboardX.SummaryWriter(logdir=filePath + "/runs/" + expName + "/train/")
        test_writer_main = tensorboardX.SummaryWriter(logdir=filePath + "/runs/" + expName+ "/test/")
        
       
        
    sfi    = torch.from_numpy(sfi).to(device)
    sfi     = sfi[None,None,:,:, :]
    
    
    points_inv   = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device), torch.from_numpy(kz.flatten()).to(device)]
    pointsI_inv   = [torch.from_numpy(wI.flatten()).to(device) ,torch.from_numpy(kyI.flatten()).to(device), torch.from_numpy(kzI.flatten()).to(device)]
    
    
   
    NxyzIm  = torch.tensor(wI.shape, device=device)
    
    ci      = torch.tensor(ci, device=device)
    
    NtFactorm      = torch.tensor(NtFactor, device=device)
  
    # Inverse operator
    op_adjm =PATinv(sf=sfi,points_inv=points_inv, pointsI_inv=pointsI_inv,c=ci, NtFactor=NtFactorm, NxyzI=NxyzIm).to(device)
    
    
    # Post processing network            
    if (netType =='resNet'):
        
        UNetm=resNetPP(n_in=1, n_out=1, width_channels=32)
    
    elif (netType =='resUnet1'):
        
        UNetm=UNetPP(n_in=1, n_out=1, width_channels=32)
        
    elif (netType =='resUnet2'):
        
        UNetm=UNetPP2(n_in=1, n_out=1, width_channels=32)    
    
    model = postProcessNetwork(op_adj=op_adjm, UNet=UNetm).to(device)
   
   
    
    loss_train = nn.MSELoss()
    loss_test = nn.MSELoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr=lValInit)
     
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter)
  
    print('Number of network parameters: ' + str(count_parameters(model)))
    
   
    tic=time.time()

    print('Start training iteration')
    
    for it in range(trainIter):
        scheduler.step() 
       
        batch = dataSet.train.next_batch(bSize)
         
         
        images = torch.from_numpy(batch[1]).float().to(device)
        projs = torch.from_numpy(batch[0]).float().to(device)
        

        model.train()   
        
        optimizer.zero_grad()
        

        
        p00, p0u = model(projs)
     
        
        if (lossFunc == 'l2_loss_1'):
            
            loss = loss_train(p0u, images)
            
        
        
        loss.backward()
        
        optimizer.step()
        
      
      
        
        if it % 25 == 0:
            if(useTensorboard):

                summaries1(train_writer_main, p0u, p00, images, loss, it, do_print=False)

            
            model.eval()
            batch = dataSet.test.next_batch(bSize)
         
         
            test_images = torch.from_numpy(batch[1]).float().to(device)
            test_projs = torch.from_numpy(batch[0]).float().to(device)
            p00Test,p0uTest = model(test_projs)
            
            if (lossFunc == 'l2_loss_1'):
            
                lossTest = loss_test(p0uTest, test_images)
            
             
             
           
            if(useTensorboard):
                
                summaries1(test_writer_main, p0uTest, p00Test, test_images,  lossTest, it, do_print=True)

    
    toc=time.time()
    
    
    print('Training took ' + str(toc-tic) + ' s' )
    
    # Save model
    
    torch.save(model, filePath + expName+'.pt')
    
    print('Training completed')
    
    

    
    return p00Test, p0uTest, test_images






def inversionNNpostProcess(dataSet,
             sfi,ky, w, kz, kyI, wI, kzI,ci,
             NtFactor,
             expName,filePath,
             netTypeInv = 'resNet',
             netTypePP = 'resNet',
             lossFunc = 'l2_loss_1',
             bSize = 1,
             trainIter = 5001,
             useTensorboard = True,
             lValInit=1e-3): 
    
              
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
  

   

    print('Computations are done using: ' + str(device))
   
        
        
    if(useTensorboard):
        
        train_writer_main = tensorboardX.SummaryWriter(logdir=filePath + "/runs/" + expName + "/train/")
        test_writer_main = tensorboardX.SummaryWriter(logdir=filePath + "/runs/" + expName+ "/test/")
        
      
    
    sfi    = torch.from_numpy(sfi).to(device)
    sfi     = sfi[None,None,:,:, :] 
    
    points_inv   = [torch.from_numpy(w.flatten()).to(device), torch.from_numpy(ky.flatten()).to(device), torch.from_numpy(kz.flatten()).to(device)]
    pointsI_inv   = [torch.from_numpy(wI.flatten()).to(device) ,torch.from_numpy(kyI.flatten()).to(device), torch.from_numpy(kzI.flatten()).to(device)]

    ci      = torch.tensor(ci, device=device)
    
   
    NtFactorm      = torch.tensor(NtFactor, device=device)
   
    NxyzIm  = torch.tensor(wI.shape, device=device)
   
   
    # Model correction network
    if (netTypeInv=='resNet'):
            
         UNetm=resNetComplex(n_in=1, n_out=1, width_channels=32).to(device)
    
    elif (netTypeInv=='resUnet'):
        
        UNetm=UNetComplex(n_in=1, n_out=1,width_channels=32, device=device).to(device)
        
    elif (netTypeInv=='resUnet2'):
        
        UNetm=UNetComplex2(n_in=1, n_out=1,width_channels=32, device=device).to(device)
        
       
    # Inverse operator
    op_adjm =PATinvNN(sf=sfi,points_inv=points_inv, pointsI_inv=pointsI_inv,c=ci, NtFactor=NtFactorm, NxyzI=NxyzIm, Unet=UNetm).to(device)
    
    
    # Post processing network            
    if (netTypePP =='resNet'):
        
        UNetPPm=resNetPP(n_in=1, n_out=1, width_channels=32)
    
    elif (netTypePP =='resUnet1'):
        
        UNetPPm=UNetPP(n_in=1, n_out=1, width_channels=32)
        
    elif (netTypePP =='resUnet2'):
        
        UNetPPm=UNetPP2(n_in=1, n_out=1, width_channels=32)    
    
    
    
    model = postProcessNetworkinvNN(op_adj=op_adjm, UNet=UNetPPm).to(device)
   
    
    loss_train = nn.MSELoss()
    loss_test = nn.MSELoss()
    
    
    optimizer = optim.Adam(model.parameters(), lr=lValInit)
   
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, trainIter)
    
    print('Number of parameters: ' + str(count_parameters(model)))
    
   
    tic=time.time()

    print('Start training iteration')
    
    for it in range(trainIter):
        scheduler.step() 
       
        batch = dataSet.train.next_batch(bSize)
         
         
        images = torch.from_numpy(batch[1]).float().to(device)
        projs = torch.from_numpy(batch[0]).float().to(device)
        

        model.train()   
        
        optimizer.zero_grad()
        
        
        
        p00, p0I,  p0u = model(projs)
     
        
        if (lossFunc == 'l2_loss_1'):
            
            loss = loss_train(p0u, images)
            
        
        
        loss.backward()
        
        optimizer.step()
        
      
      
        
        if it % 25 == 0:
            if(useTensorboard):

                summaries1(train_writer_main, p0u, p00, images, loss, it, do_print=False)

            
            model.eval()
            batch = dataSet.test.next_batch(bSize)
         
         
            test_images = torch.from_numpy(batch[1]).float().to(device)
            test_projs = torch.from_numpy(batch[0]).float().to(device)
            p00Test, p0ITest, p0uTest = model(test_projs)
            
            if (lossFunc == 'l2_loss_1'):
            
                lossTest = loss_test(p0uTest, test_images)
            
             
            
            if(useTensorboard):
                
                summaries1(test_writer_main, p0uTest, p00Test, test_images,  lossTest, it, do_print=True)

    

    toc=time.time()
    
    print('Training took ' +str(toc-tic)+ ' s')
    
    # save model
    torch.save(model, filePath + expName+'.pt')
    
    print('Training completed')

    
    
    return p00Test, p0ITest, p0uTest, test_images




#########################################################################



def EVALinversionNNpostProcess(dataSet,
             expNameNN,filePathNN,
             saveResultsFlag,expName,filePath): 
    
    
    print('Start evaluation')
                   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
  
    print('Computations are done using: ' + str(device))
  
    
  
    model = torch.load(filePathNN + expNameNN +'.pt')
    
    
    ind=dataSet.test.ind
    
    
    batch = dataSet.test.selected_set(ind)
         
         
    test_images_all = torch.from_numpy(batch[1]).float().to(device)
    test_projs_all = torch.from_numpy(batch[0]).float().to(device)
     
    for it2 in range(len(ind)):
        
        model.eval() 
        
        ind2=ind[it2]
        
        test_images=test_images_all[it2]
        test_projs=test_projs_all[it2]
        
        test_images=test_images.unsqueeze(0)
        test_projs=test_projs.unsqueeze(0)
        
        tic=time.time()
        
        p00Test, p0ITest,  p0uTest = model(test_projs)
         
        toc=time.time()
        
        print(' Reconstruction took  ' +str(toc-tic) + ' s')
        
    
        p00Test = p00Test.cpu().detach().numpy()
    
        p0ITest = p0ITest.cpu().detach().numpy()
    
        p0uTest = p0uTest.cpu().detach().numpy()
    
        
        
        
        test_images = test_images.cpu().detach().numpy()
        test_projs = test_projs.cpu().detach().numpy()
        
      
    
    
    
        # save results?

        if saveResultsFlag:
        	# FFT reconstruction
        	np.save(filePath+expName+'_p00Test_'+ str(ind2), p00Test)
        
        	# FFT reconstruction + model correction
        	np.save(filePath+expName+'_p0ITest_'+ str(ind2), p0ITest)
        
        	#FFT reconstruction + model correction +post processing
        	np.save(filePath+expName+'_p0uTest_'+ str(ind2), p0uTest)
        
        
        	# true initial pressure
        	np.save(filePath+expName+'_true_'+ str(ind2), test_images)

        	# pressure data
        	np.save(filePath+expName+'_projs_'+ str(ind2), test_projs)
        
    
    if saveResultsFlag:    
    	np.save(filePath+expName+'_ind', ind)
    
    
    return p00Test, p0ITest, p0uTest, test_images



def EVALpostProcess(dataSet,
             expNameNN,filePathNN,
             saveResultsFlag, expName,filePath): 
    
    
    print('Start evaluation')
          
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')        
  
    print('Computations are done using: ' + str(device))
    
    model = torch.load(filePathNN + expNameNN +'.pt')
    
    
    ind=dataSet.test.ind
    
    
    batch = dataSet.test.selected_set(ind)
         
         
    test_images_all = torch.from_numpy(batch[1]).float().to(device)
    test_projs_all = torch.from_numpy(batch[0]).float().to(device)
   
    
    for it2 in range(len(ind)):
        
        model.eval() 
        
        ind2=ind[it2]
        
        test_images=test_images_all[it2]
        test_projs=test_projs_all[it2]
        
        test_images=test_images.unsqueeze(0)
        test_projs=test_projs.unsqueeze(0)
        
        tic=time.time()
        
        p00Test, p0uTest = model(test_projs)
        
        toc=time.time()    
        
        print(' Reconstruction took  ' +str(toc-tic) + ' s')
    
        p00Test = p00Test.cpu().detach().numpy()
    
         
        p0uTest = p0uTest.cpu().detach().numpy()
    
        
        
        
        test_images = test_images.cpu().detach().numpy()
        test_projs = test_projs.cpu().detach().numpy()
        
      
    
    
    
        # save results?
        if saveResultsFlag:
        
        
        	# FFT reconstruction
        	np.save(filePath+expName+'_p00Test_'+ str(ind2), p00Test)
        
        	# FFT reconstruction + post processing
        	np.save(filePath+expName+'_p0uTest_'+ str(ind2), p0uTest)
        
    
        
        	# true initial pressure
        	np.save(filePath+expName+'_true_'+ str(ind2), test_images)
    
        	# pressure data
        	np.save(filePath+expName+'_projs_'+ str(ind2), test_projs)
        
    
        
    if saveResultsFlag:
        np.save(filePath+expName+'_ind', ind)    
    
    return p00Test, p0uTest, test_images
