'''

@author: mjiang,jgirard

'''

import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import scipy.ndimage.filters as pflt
import scipy.fftpack as pfft
import scipy.signal as psg
from scipy import special
from scipy import ndimage
import pyfits as fits
import pylab
# import sparse2d
import param as pm
from math import *
    
 
def init():
    pm.trTab = []
    pm.trHead= ''
    
def mad(alpha):
    dim = np.size(np.shape(alpha)) 
    if dim == 1:
        alpha = alpha[np.newaxis,np.newaxis,:]
    elif dim == 2:
        alpha = alpha[np.newaxis,:,:]
#         (nx,ny) = np.shape(alpha)
#         alpha = alpha.reshape(1,nx,ny)
    (nz,nx,ny) = np.shape(alpha)
    sigma = np.zeros(nz)
    for i in np.arange(nz):
        sigma[i] = np.median(np.abs(alpha[i,:,:] - np.median(alpha[i,:,:]))) / 0.6745
    alpha = np.squeeze(alpha)        
    return sigma

def mad_norm(alpha,noise):
    (nz,nx,ny) = np.shape(alpha)
    for i in np.arange(nz):
        if noise[i] != 0:
            alpha[i,:,:] /= noise[i]
        
def mad_unnorm(alpha,noise):
    (nz,nx,ny) = np.shape(alpha)
    for i in np.arange(nz):
        if noise[i] != 0:
            alpha[i,:,:] *= noise[i]

def softTh(alpha,th,weights=None,reweighted=False):
    dim = np.size(np.shape(alpha))        
    if dim == 2:
        (nx,ny) = np.shape(alpha)
        alpha = alpha.reshape(1,nx,ny)
    (nz,nx,ny) = np.shape(alpha)
    if np.size(th) == 1:
        thTab = np.ones(nz) * th
    else:
        thTab = th
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i,:,:])[np.abs(alpha[i,:,:])<=thTab[i]] = 0
            (alpha[i,:,:])[alpha[i,:,:]>0] -= thTab[i]
            (alpha[i,:,:])[alpha[i,:,:]<0] += thTab[i]
        else:
            (alpha[i,:,:])[np.abs(alpha[i,:,:])<=thTab[i]*weights[i,:,:]] = 0
            (alpha[i,:,:])[alpha[i,:,:]>0] -= (thTab[i]*weights[i,:,:])[alpha[i,:,:]>0]
            (alpha[i,:,:])[alpha[i,:,:]<0] += (thTab[i]*weights[i,:,:])[alpha[i,:,:]<0]
            
    if dim == 2:
        alpha = alpha.reshape(nx,ny)

def hardTh(alpha,th,weights=None,reweighted=False):
    dim = np.size(np.shape(alpha)) 
    if dim == 2:
        (nx,ny) = np.shape(alpha)
        alpha = alpha.reshape(1,nx,ny)
    (nz,nx,ny) = np.shape(alpha)
    if np.size(th) == 1:
        thTab = np.ones(nz) * th
    else:
        thTab = th
    for i in np.arange(nz):
        if not reweighted:
            (alpha[i,:,:])[np.abs(alpha[i,:,:])<=thTab[i]] = 0
        else:
            (alpha[i,:,:])[np.abs(alpha[i,:,:])<=thTab[i]*weights[i,:,:]] = 0
    if dim == 2:
        alpha = alpha.reshape(nx,ny)

def softTh2d1d(alpha,thTab,weights=None,reweighted=False):
    nz = np.size(thTab)
#     print weights.shape
#     print thTab.shape
    for i in np.arange(nz):
        if reweighted:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0
            (alpha[i])[alpha[i]>0] -= (thTab[i]*weights[i])[alpha[i]>0]
            (alpha[i])[alpha[i]<0] += (thTab[i]*weights[i])[alpha[i]<0]
        else:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
            (alpha[i])[alpha[i]>0] -= thTab[i]
            (alpha[i])[alpha[i]<0] += thTab[i]
            
            
def hardTh2d1d(alpha,thTab,weights=None,reweighted=False):
    nz = np.size(thTab)
    for i in np.arange(nz):
        if reweighted:
            (alpha[i])[np.abs(alpha[i])<=thTab[i]*weights[i]] = 0
        else:
            (alpha[i])[abs(alpha[i])<=thTab[i]] = 0
            
    
def spectralNorm(nx,ny,nz,Niter,tol):
    normalization = False
    stWav = sparse2d.Starlet2D(nx,ny,nz)
    matA = np.random.randn(nx,ny)
    spNorm = LA.norm(matA)
    matA /= spNorm
    it = 0
    err = 10*abs(tol)
    while it < Niter and err > tol:
#         wt = star2d(matA,nz)
#         matA = adstar2d(wt)  
        # Boost version
        wt = stWav.transform_gen1(matA,normalization)
        matA = stWav.trans_adjoint_gen1(wt,normalization)
        spNorm_new = LA.norm(matA)
        matA /= spNorm_new
        err = abs(spNorm_new - spNorm)/spNorm_new
        print "Iteration:"+str(it)+"Error:"+str(err)
        spNorm = spNorm_new
        it += 1       
    return spNorm


     
   
                
        
        
    
    
        
    
