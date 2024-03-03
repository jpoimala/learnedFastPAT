#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  5 11:21:47 2021

@author: jtick20
"""



import matplotlib.pyplot as plt

import kgrids_pytorch_kspace_3D as kgrids

import nets_pytorch_kspace_3D as PAT
import h5py
import numpy as np

# Reconstruction speed of sound
cList=(np.array([1500]))


# Initial learning rate
iValList=(np.array([1e-2]))


for it2 in range(len(cList)):
    
            
    
    for it3 in range(len(iValList)):
        
        # simulation speed of sound
         
        ctest='1400_1600' 
        ctrain='1400_1600'
        
        # recon speed of sound
        c=cList[it2] # speed of sound [m/s] (homogeneous)
        

        
        NtFactor=5 # more time points
        
        
        
        # #Load training and test sets
        # mat-files including variables 'imagesTrue' and 'dataTrue'
        # that are indexed  imagesTrue(z position, y position, x position, sample)
        # and  dataTrue(sensor z position, sensor y position, time step, sample).
        
       
        trainSet = '.../trainDataSet_vessel_3D_' +str(NtFactor) +'xtimesteps' + '_c_'  + ctrain + '.mat'
        testSet = '.../testDataSet_vessel_3D_' +str(NtFactor) +'xtimesteps' + '_c_'  + ctest + '.mat'
          
        # Indices of samples for which results are computed
        trainInd=[0]
        #testInd=(np.array([2,4, 9, 22, 26])-1)
        testInd=(np.arange(45))
        
        
        dataSet = PAT.read_data_sets(trainSet,testSet, trainInd, testInd)
    
        
        
        # kgrid parameters
        dx=5.3e-5*2
        dy=5.3e-5*2  # spatial step [m]
        dz=5.3e-5*2
        
    
        dt=3*5/3*1e-8  # time step
       
        

        kgridsRecon=kgrids.kspaceKGrids(dx,dy, dz, dt,c, NtFactor)
        
        
        
        
        # #########################################################################
        # # Post process
        # # Training parameters
        bSize               = int(1)
        trainIter           = 50001           # Training iteration
        useTensorboard      = True          # Use Tensorboard for tracking
        filePath            = '.../Results/'       # Where to save the network
        lValInit            = iValList[it3] #Initial learning rate
        lossFunc    = 'l2_loss_1'        #'l2_loss_1
        netType    = 'resUnet2'         # 'resUnet1, resUnet2 , resNet'
        
        saveResultsFlag = True	# save results?

        
        testName           = 'vessel_c' + ctest +'_train_vessel_c' + ctrain + '_cRecon_' + str(c) +   '_pp' # test name
        
        experimentName =  testName + '_' + netType  + '_' + lossFunc +'_it' + str(trainIter) + '_iVal' + str(lValInit) + '_bS' + str(bSize) + '_c_' +str(c) + '_' + str(NtFactor) + 'xtimesteps' #Name for this experiment
        
   
       
            
        [sfi,kyi, wi, kzi, kyiI, wiI, kziI]=kgridsRecon.inverse(dataSet.test.data[0,0])
          
        sfi=np.float32(sfi)
        wi=np.float32(wi)
        kyi=np.float32(kyi)
        kzi=np.float32(kzi)
            
        wiI=np.float32(wiI)
        kyiI=np.float32(kyiI)
        kziI=np.float32(kziI)
            
            
        
        # Train FFT + post processing network
        
        p00PP, p0PP, p0origPP, = PAT.postProcess(dataSet,
                      sfi,kyi, wi, kzi, kyiI, wiI, kziI, c,
                      NtFactor,
                      experimentName,filePath,
                      netType=netType,
                      lossFunc = lossFunc,
                      bSize = bSize,
                      trainIter = trainIter,
                      useTensorboard = useTensorboard,
                      lValInit=lValInit)
            
            
    
            
        # Get results for FFT + post processing
        
        
        
        p0, p0PP, p0orig, = PAT.EVALpostProcess(dataSet,
                      experimentName,filePath,
                      saveResultsFlag,experimentName,filePath)
            
             
            
        #    #####################
        # # plot results
            
        fig, axs =plt.subplots(3,3)
        axs[0,0].imshow(p0orig[0,0].max(0))
        axs[0,0].set_title('True')
        axs[1,0].imshow(p0orig[0,0].max(1))
        axs[2,0].imshow(p0orig[0,0].max(2))
            
            
        axs[0,1].imshow(p0[0,0].max(0))
        axs[0,1].set_title('FFT')
        axs[1,1].imshow(p0[0,0].max(1))
        axs[2,1].imshow(p0[0,0].max(2))
            
            
            
        axs[0,2].imshow(p0PP[0,0].max(0))
        axs[0,2].set_title('FFT+PP')
        axs[1,2].imshow(p0PP[0,0].max(1))
        axs[2,2].imshow(p0PP[0,0].max(2))
            
            
           
     