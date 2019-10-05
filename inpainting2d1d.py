'''
PySASIR2D1D

@author: mjiang, jgirard
'''
from util import *
from wav1d import *
from wav2d import *
from wav2d1d import *
from algo2d1d import *
import numpy as np
import pyfits as fits
import matplotlib.pyplot as plt
import pylab
import time
import os


ind = 1
  
def do2D1D(filecubemask,filecubedirty,jobname,positivity=True):
    drResult = 'results-'+jobname+'/'

    mask = fits.getdata(filecubemask)
    maskSh = ifftshift2d1d(mask)
    psf=np.real(fftshift2d1d(ifft2d1d(maskSh)))
    tab_psf=np.max(psf,axis=(1,2))
    cubedirty=fits.getdata(filecubedirty).astype(np.float64)
    cubefftDeg = fft2d1d(cubedirty)

# Code input
    FlagPositivity=positivity
    scale2d=4 #4
    scale1d=4
    w2dname='starlet' #'direct' #'starlet'
    w1dname='9/7' #'haar' #'9/7'
    Nsig=2.8
    mu=1.0
    sigma=0
    Niter=200  # initial loop number softthreshold
    Miter=50  # loop in reweigthed 
    Liter=100 # hard threshold at the end
    reweightedTimes=4
    Err=1e-6
#################
    coarse=False
    KillLastScale=True
    reweighted=True
    normIRF=False #True
    NsigDec=True #True
    NsigFst=True #True
    thMode='soft'            
    
    ParDict={"FlagPositivity":FlagPositivity, "scale2d":scale2d, "scale1d":scale1d,"w2dname":w2dname,"w1dname":w1dname,"Nsig":Nsig,\
"mu":mu,"sigma":sigma,"Niter":Niter,"Miter":Miter,"Liter":Liter,\
"reweightedTimes":reweightedTimes,"Err":Err,"coarse":coarse,\
"reweighted":reweighted,"normIRF":normIRF,"NsigDec":NsigDec,\
"NsigFst":NsigFst,"thMode":thMode,"filecubemask":filecubemask,\
"filecubedirty":filecubedirty,"jobname":jobname,"drResult":drResult}

    start = time.clock() 
            # cube_rec = ISTA2d1d(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Err,maskSh,FISTA=True,Positivity=True,coarse=False,reweighted=True,normIRF=False,NsigDec=False,NsigFst=False,thMode='soft',Fourier=True)
            # cube_rec = ISTA2d1d_boost(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Err,maskSh,FISTA=True,Positivity=True,coarse=False,reweighted=True,normIRF=False,NsigDec=False,NsigFst=False,thMode='soft',Fourier=True)
            # cube_rec = ISTA2d1d_Analysis(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Err,maskSh,FISTA=True,Positivity=True,coarse=False,reweighted=True,normIRF=False,NsigDec=False,NsigFst=False,thMode='soft',Fourier=True)
            # cube_rec = ISTA2d1d_Analysis_boost(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Err,maskSh,FISTA=True,Positivity=True,coarse=False,reweighted=True,normIRF=False,NsigDec=False,NsigFst=False,thMode='soft',Fourier=True)
            # cube_rec = GFrBk2d1d_Analysis(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Err,maskSh,coarse=False,reweighted=True,normIRF=False,NsigDec=True,NsigFst=False,thMode='soft',Fourier=True)
    #         cube_rec = vu2d1d_Analysis(cubefftDeg,scale2d,scale1d,w1dname,Nsig,mu,sigma,Niter,Miter,Err,maskSh,coarse=True,reweighted=True,normIRF=True,NsigDec=True,NsigFst=True,thMode='soft')
    cube_rec,rsd,noise,errTab,weights = vu2d1d_Analysis_new(cubefftDeg,scale2d,scale1d,w2dname,w1dname,Nsig,mu,sigma,Niter,Miter,Liter,reweightedTimes,Err,maskSh,Positivity=FlagPositivity,coarse=coarse,reweighted=reweighted,normIRF=normIRF,NsigDec=NsigDec,NsigFst=NsigFst,thMode=thMode,tab_psf=tab_psf,KillLastScale=KillLastScale)
    if not os.path.exists(drResult):
        os.makedirs(drResult)
    fits.writeto(drResult+'Vu_Ana'+str(ind)+'.fits',cube_rec,clobber=True)
    fits.writeto(drResult+'Cube_residuals'+str(ind)+'.fits',np.real(rsd),clobber=True)
    fits.writeto(drResult+'noise'+str(ind)+'.fits',noise,clobber=True)
    fits.writeto(drResult+'errorVu_Ana'+str(ind)+'.fits',errTab,clobber=True)
    fits.writeto(drResult+'weights'+str(ind)+'.fits',weights,clobber=True)
    fits.writeto(drResult+'Dirty'+str(ind)+'.fits',cubedirty,clobber=True)
    np.savez(drResult+'Parset.npz',ParDict=ParDict)
    end = time.clock()
    print end-start
    print "Reconstruction ended"
    return cube_rec

