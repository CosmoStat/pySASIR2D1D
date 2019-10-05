'''

@author: mjiang,jgirard
'''
import pdb
import progressbar
import numpy as np
from numpy import linalg as LA
import scipy.fftpack as pfft
import scipy.signal as psg
import pyfits as fits
import pylab
# import sparse
# import sparse2d
import param as pm
from util import *
from wav2d1d import *
import time

def vu2d1d_Analysis_new(vsb,scale2d,scale1d,w2dname,w1dname,Nsig,tau,sigma,Niter,Miter,Liter,reweightedTimes,Err,mask,Positivity,coarse=False,reweighted=False,normIRF=False,NsigDec=True,NsigFst=False,thMode='soft',tab_psf=None,KillLastScale=False):
    (bd,nx,ny) = np.shape(vsb)
    gen2 = False                                         # Using 2nd generation of starlet or not 
    if w2dname == 'starlet':
        head = 'star2d_gen2' if gen2 else 'star2d_gen1'     # Store the transformation head and calculate impulse response of transformation
        trans = 1 if gen2 else 2
        if (normIRF or sigma != 0) and pm.trHead != head:
            pm.trTab = nsNorm(nx,ny,scale2d,trans)
            pm.trHead = head
    elif w2dname == 'direct':
        scale2d=1
    if w1dname == 'haar':
        wtype = 1
    elif w1dname == '9/7':
        wtype = 2
    
    
        ############## Mask simulation ########################
    simuNoise = np.random.randn(bd,nx,ny)
    simuNoise_FT = fft2d1d(simuNoise)
    simuNoise_FT_m_inv = np.real(ifft2d1d(mask*simuNoise_FT))
#     simuNoise_FT_inv = np.real(ifft2d1d(simuNoise_FT))
    simuNoise_FT_m_inv_wt,b_simuNoise_FT_m_inv = dec2d1d(simuNoise_FT_m_inv,scale2d,scale1d,gen2=gen2,w2dname=w2dname,w1dname=w1dname,wtype=wtype)
#     simuNoise_FT_inv_wt,b_simuNoise_FT_inv = dec2d1d(simuNoise_FT_inv,scale2d,scale1d,gen2=gen2,wname=w1dname,wtype=wtype)
    
#    bookkeeping = map(np.int,b_simuNoise_FT_m_inv) #JGMOD
    bookkeeping = b_simuNoise_FT_m_inv.astype(np.int) #JGMOD
    nz = np.size(simuNoise_FT_m_inv_wt,axis=0)
    print "POST RANDOM"
#    print "bookkeeping"
#    print bookkeeping,nz
    if w2dname == 'starlet':
        lenwt1d = nz/scale2d
    else:    #JGMOD
        lenwt1d = bookkeeping[0]        #JGMOD
    sigma_glob = np.std(simuNoise_FT_m_inv_wt.flatten())
    noiseNormTable_m = np.std(simuNoise_FT_m_inv_wt,axis=(1,2))
#     noiseNormTable = np.std(simuNoise_FT_inv_wt,axis=(1,2))
#     maskNormTable = noiseNormTable/noiseNormTable_m
#     maskNormTable_mean1 = np.mean(maskNormTable[-lenwt1d])  
    
    errTab = np.zeros(Niter+Miter*reweightedTimes+Liter)                        # Table used to store result error per iteration
    #Initialization
#     x = np.real(ifft2d1d((vsb)))

    x = np.zeros((bd,nx,ny))

    #pdb.set_trace()

    if coarse:
        print "coarse"
        print bookkeeping[0]
	print bookkeeping
        u = np.zeros((nz-np.int(bookkeeping[0]),nx,ny))
    else:
        print "nocoarse"
        u = np.zeros((nz,nx,ny)) 

#     (c_u,b_u) = wavOrth1d(x[:,0,0],scale1d,wname=w1dname,wtype=wtype)               # Save computing time for 1d band information b_u
#     u,b_u = dec2d1d(x,scale2d,scale1d,gen2=gen2,wname=w1dname,wtype=wtype)
    #fits.writeto('u'+str(it)+'.fits',u,clobber=True) 
    # Weights matrix for the l1 reweighted

    if reweighted:
        weights = np.ones((nz-np.int(bookkeeping[0]),nx,ny)) if coarse else np.ones((nz,nx,ny))
    else:
        weights = None
        Miter = 0
        Liter = 0
    # Threshold decreases    
    if NsigDec:
        NsigInit = Nsig+2 
    
    
    err = abs(Err)
    it = 0
    sigmaVu = 1.0                                         # Parameter for the sparstiy costraint
    reweightedCount = 0                                               # Iterations for multi-scale positivity constraint
    var_rsd = 0
#     NsigHard = 3                                            # Threshold for hard-thresholding

    
    print "Analysis optimization using Vu method"
    print
        


    bar = progressbar.ProgressBar(redirect_stdout=True,max_value=Niter+Miter*reweightedTimes+Liter)
    while it<(Niter+Miter*reweightedTimes+Liter) and err >= Err:
	t0=time.time()
	itmax=Niter+Miter*reweightedTimes+Liter
        print "ITERATION %s/%s"%(str(it),str(itmax))
        
        if coarse:
            u_ = np.concatenate((np.zeros((bookkeeping[0].astype(np.int32),nx,ny)),u),axis=0)
        else:
            u_ = u
        fits.writeto('ubefore'+str(it)+'.fits',u_,clobber=True) 
	if KillLastScale:
	    print "Killing thinner scales"
	    print u_.shape
	    print bookkeeping
	    u_[bookkeeping[-1]:,:,:]=0
        fits.writeto('uafter'+str(it)+'.fits',u_,clobber=True)         
        termP = rec2d1d(u_,bookkeeping,bd,gen2=gen2,adjoint=True,w2dname=w2dname,w1dname=w1dname,wtype=wtype)


        rsd = (vsb - mask*fft2d1d(x))
	#print "vsb" #JGMOD
	#print vsb #JGMOD
	#print "mask" #JGMOD
	#print mask #JGMOD
	#print "x" #JGMOD
	#print x #JGMOD
        var_rsdN = np.var(rsd)
        rsd1 = np.real(ifft2d1d(mask*rsd))
        
#         fits.writeto('rsd'+str(it)+'.fits',np.real(rsd1),clobber=True)
        rsdTr,b_rsdTr = dec2d1d(rsd1,scale2d,scale1d,gen2=gen2,w2dname=w2dname,w1dname=w1dname,wtype=wtype)

        if sigma == 0:
            if normIRF:
                if w2dname == 'starlet':
                    sigmaEval=np.zeros(lenwt1d) 
                    fstScaleWt = rsdTr[-lenwt1d:]
        #             fstScaleWt = np.reshape(fstScaleWt,(np.size(fstScaleWt)))
                    sigmaEval = mad(fstScaleWt)
                    noise = np.zeros(nz)
                    for fm in np.arange(lenwt1d):
        #                 sigmaEval[fm] = sigmaEval[fm]*maskNormTable_mean1
                        noise[fm::lenwt1d] = sigmaEval[fm] * noiseNormTable_m[fm::lenwt1d]/noiseNormTable_m[-lenwt1d+fm]
                elif w2dname == 'direct':
                    rsdTr1 = (rsdTr[np.int(bookkeeping[0]):]).flatten()
                    sigmaEval = mad(rsdTr1)
                    noise = sigmaEval*noiseNormTable_m/sigma_glob
                            
            else:
                noise = mad(rsdTr)
        else:
            noisefm = sigma * pm.trTab[::-1]
            noise = np.repeat(noisefm,nz/scale2d)
        
        xn = x - tau * (termP - rsd1)
	#print "x" #JGMOD
	#print x #JGMOD
	#print "termP" #JGMOD
	#print termP #JGMOD
	#print "rsd1" #JGMOD
	#print rsd1 #JGMOD 
        #pdb.set_trace()
        if Positivity:
            xn[xn<0] = 0 
             
        termQ1,b_termQ1 = dec2d1d(2*xn-x,scale2d,scale1d,gen2=gen2,w2dname=w2dname,w1dname=w1dname,wtype=wtype)
        
        if coarse:
            termQ1_wt = np.copy(termQ1[np.int(bookkeeping[0]):])
        else:
            termQ1_wt = termQ1
            
        termQ1_wt *= sigmaVu
        termQ1_wt += u
        termQ2 = np.copy(termQ1_wt)
            
        if NsigDec:
            if it < Niter:
                th = NsigInit - (NsigInit - Nsig)*it/(Niter-1)
            else:
                th = Nsig
            if th < Nsig:
                th = Nsig
            print th
        else:
            th = Nsig
        thTab = np.ones_like(noise) * th
#         if reweighted and it > Niter-1+reweightedTimes*Miter:
#             thTab = NsigHard*noise
#         else:
#             thTab = th * noise
            
        if NsigFst:
            if w2dname == 'starlet':
                thTab[-lenwt1d:] += 1
        
        thTab *= sigmaVu
        
        if coarse:
            thTab = np.delete(thTab,np.arange(bookkeeping[0]))
            noise = np.delete(noise,np.arange(bookkeeping[0]))
            
        mad_norm(termQ2,noise)
#             thTab[:bookkeeping[0]] = 0                                # Don't treat the 2d-1d approximation scale
              
        
#         fits.writeto('p'+str(it)+'.fits',p,clobber=True)
        # Positivity constraint
        
#         pWt,b_pWt = dec2d1d(p,scale2d,scale1d,gen2=gen2,wname=w1dname,wtype=wtype)
#         pWt[pWt<0] = 0
#         fits.writeto('termQ2_'+str(it)+'.fits',termQ2,clobber=True)
        
        if thMode == 'soft':
            if reweighted and it > Niter-1+reweightedTimes*Miter:
                weights[weights<1] = 0
                print "Hardthresholding"
                softTh2d1d(termQ2,thTab,weights,reweighted=True)
            elif reweighted and it > Niter-1 and it <= Niter-1+reweightedTimes*Miter:
                softTh2d1d(termQ2,thTab,weights,reweighted=True)               
            else:
                softTh2d1d(termQ2,thTab)
        mad_unnorm(termQ2,noise)
        u = termQ1_wt - termQ2
         
#         fits.writeto('xWt'+str(it)+'.fits',xWt,clobber=True)
        # Using reweighted l1-norm 
        if reweighted and it-Niter==reweightedCount*Miter and reweightedCount<reweightedTimes:  
            xWt,b_xWt = dec2d1d(xn,scale2d,scale1d,gen2=gen2,w2dname=w2dname,w1dname=w1dname,wtype=wtype)
	    #pdb.set_trace()
            weightsN = updateWeights(xWt[bookkeeping[0]:],noise,Nsig,wav2d1dNorm=False,scale2d=scale2d,gen2=gen2)
            weights *= weightsN
            reweightedCount += 1
            print 'Change weights at iteration:'+str(it)
            print reweightedCount

        # Current error
#         num = (abs(xn - x)).max()
#         denum = (abs(xn)).max()
        err = abs(var_rsdN-var_rsd)
	print var_rsdN #JGMOD
	print var_rsd #JGMOD 
        var_rsd = var_rsdN  
        errTab[it] = err 
        it += 1
        print "Iteration:"+str(it)
        print "Current error:" + str(err)
        x = np.copy(xn)
        tf=time.time()
	print "Iteration time = ",tf-t0 
#     ax = np.arange(1,Niter+1)
#     plt.figure('Error evolution of Vu_Ana')  
#     plt.plot(ax,errTab)
#     plt.figure('Evolution of noise estimation of ISTA')
#     for ns in np.arange(len(nsTab[:,0])):
#         plt.plot(ax,nsTab[ns,:],pm.colors[ns]+pm.lineStyles[1])
#     plt.figure(),plt.plot(sigmaEval)



	rsd = ifft2d1d(vsb - mask*fft2d1d(x))
	fits.writeto('recit'+str(it)+'.fits',np.real(x),clobber=True)
	fits.writeto('resit'+str(it)+'.fits',np.real(rsd),clobber=True)
        bar.update(it)

    rsd = ifft2d1d(vsb - mask*fft2d1d(x))
    for fm in np.arange(bd):
        x[fm] *= tab_psf[fm]
    
#     fits.writeto('results/noise'+str(pm.count6)+'.fits',noise,clobber=True)
#     fits.writeto('results/weights'+str(pm.count6)+'.fits',weights,clobber=True)
#     xWt,b_xWt = dec2d1d(x,scale2d,scale1d,gen2=gen2,wname=w1dname,wtype=wtype)
#     coeff = xWt.flatten()
#     coeffSort = np.sort(abs(coeff))
#     coeffSort = coeffSort[::-1]
#     plt.figure('Sorted 2d1d transform coefficients via Vu method')
#     plt.plot(coeffSort)
   
#     fits.writeto('results/Vu_Ana'+str(pm.count6)+'.fits',x,clobber=True)
#     fits.writeto('results/errorVu_Ana'+str(pm.count6)+'.fits',errTab,clobber=True)
#     fits.writeto('noiseISTA'+str(pm.count1)+'.fits',nsTab,clobber=True)
#     fits.writeto('results/coefficientsVu_Ana'+str(pm.count6)+'.fits',coeffSort,clobber=True)
#     pm.count6 += 1  

#     fig, axarr = plt.subplots(3, 2)
#     fig.suptitle(r'Vu method reconstruction, $\sigma_n$=0, mask$\approx$20%',fontsize=15)
#     im1 = axarr[0, 0].imshow(x[0])
#     axarr[0, 0].set_title('Frame 1')
#     divider1 = make_axes_locatable(axarr[0, 0])
#     cax1 = divider1.append_axes("right", size="10%", pad=0.05)
#     fig.colorbar(im1,cax=cax1)
#      
#     im2 = axarr[0, 1].imshow(x[31])
#     axarr[0, 1].set_title('Frame 31')
#     divider2 = make_axes_locatable(axarr[0, 1])
#     cax2 = divider2.append_axes("right", size="10%", pad=0.05)
#     fig.colorbar(im2,cax=cax2)
#      
#     axarr[1, 0].stem(x[:,7,25])
#     axarr[1, 0].set_title('Profile of transient')
#     axarr[1, 1].stem(x[:,16,16])
#     axarr[1, 1].set_title('Profile of constant source')
#     axarr[2, 0].stem(xWt[:,7,25])
#     axarr[2, 0].set_title('Coefficients of transient')
#     axarr[2, 1].stem(xWt[:,16,16])
#     axarr[2, 1].set_title('Coefficients of constant source')
        
    return (x,rsd,noise,errTab,weights)
