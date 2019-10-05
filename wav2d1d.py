'''

@author: mjiang,jgirard
'''
import pdb  #JGMOD
from util import *
from wav1d import *
from wav2d import *
import pyfits as fits
from numba import jit,njit,prange,double,int32
import time #JGMOD
# from sparse import *
import scipy.fftpack as pfft
# import sparse2d

def calculateSNR(dirtyCube,patch):
    '''
    Return the SNR of the dirty image
    
    @param dirtyCube: dirty image/cube of which the SNR should be calculated
    
    @param patch: Defined zone for the source
    
    @return: SNR on decible of the dirty image
    '''
    nz,nx,ny = np.shape(dirtyCube)
    source1 = dirtyCube[:,nx/2-patch/2:nx/2+patch/2+1,ny/2-patch/2:ny/2+patch/2+1]
    print (nx/2-patch/2,nx/2+patch/2+1,ny/2-patch/2,ny/2+patch/2+1)
    average_source1 = np.mean(source1)      # The average of signal
    backgroundMask = np.ones((nz,nx,ny))
    backgroundMask[:,nx/2-patch/2:nx/2+patch/2+1] = 0
    background = dirtyCube * backgroundMask
    nbPxl_background = (nz*nx*ny-nz*patch*patch)
    average_background = np.sum(background)/nbPxl_background
    std_background = np.sqrt(np.sum((background - average_background*backgroundMask)**2)/nbPxl_background)
    snrDirty = 20*np.log10(average_source1/std_background)
    return snrDirty
    

def SNR2Dirty(cube,mask,snr=10):
    '''
    Return the dirty image/cube and the deviation of gaussien noise injected on the UV-plan accroding to the SNR given
    
    @param cube: Ground-truth dataset
    
    @param mask: UV-plan mask corresponding to the distribution of antennae
    
    @param snr: The given SNR, defaut is 10. SNR = 20log(mu_signal/sigma_background) 
    https://en.wikipedia.org/wiki/Signal-to-noise_ratio_(imaging)
    
    @return: (The dirty image/cube, the deviation of noise)
     
    '''
    nz,nx,ny = np.shape(cube)
    if np.shape(cube) != np.shape(mask):
        print "The size of mask does not match that of the dataset"
    patch = 3                               # Definition of central resolved source of zone patch*patch, 
                                            # patch is an odd number
    cubefft = fft2d1d(cube)
    cubefftDeg = cubefft * mask
    dirtyCube = np.real(ifft2d1d(cubefftDeg))        # Dirty cube without injection of noise
    fits.writeto('maskSh.fits',mask,clobber=True)
    fits.writeto('dirtyCube.fits',dirtyCube,clobber=True)
    snr0 = calculateSNR(dirtyCube,patch)
    print "The initial SNR = "+str(snr0)
    sigma = 0
    dsigma = 0.1
    dirOld = 1
    dirNew = 1
    if snr0 <= snr:
        print "The SNR given is so high that can't inject the noise"
    else:
        iter = 1
        while abs(snr0-snr) >= 1e-3:
            sigma += dirOld*dsigma
            n1 = sigma*sqrt(nx*ny)/sqrt(2)*np.random.randn(nz,nx,ny)
            n2 = sigma*sqrt(nx*ny)/sqrt(2)*np.random.randn(nz,nx,ny)
            print "Iteration"+str(iter)+":"
            print sigma
            cubefftDeg_n = (cubefft + n1 + n2*1j)*mask    
            dirtyCube = np.real(ifft2d1d(cubefftDeg_n))                      # Dirty cube with injection of noise
            snr0 = calculateSNR(dirtyCube,patch)            
            print snr0
            dirNew = np.sign(snr0 - snr)
            if dirNew != dirOld:
                dsigma /= 2
            dirOld = dirNew
            iter += 1
    return (dirtyCube,sigma)
    


def dirtyCube(cube,mask,noise=False,sigma=1.0):
    (nz,nx,ny) = np.shape(cube)
    cubeUV = fft2d1d(cube)
    if noise:
        n = sigma * np.random.randn(nz,nx,ny)
        cubeUV += n
    cubeUVMask = cubeUV*mask    
    dirtyCube = ifft2d1d(cubeUVMask)
    return dirtyCube

def fftshift2d1d(cubefft):
    (nz,nx,ny) = np.shape(cubefft)
    cubefftSh = np.zeros_like(cubefft) + np.zeros_like(cubefft) * 1j        # Convert to complex array
    for fm in np.arange(nz):
        cubefftSh[fm] = pfft.fftshift(cubefft[fm])
    return cubefftSh

def ifftshift2d1d(cubefftSh):
    (nz,nx,ny) = np.shape(cubefftSh)
    cubefft = np.zeros_like(cubefftSh) + np.zeros_like(cubefftSh) * 1j           # Convert to complex array
    for fm in np.arange(nz):
        cubefft[fm] = pfft.ifftshift(cubefftSh[fm])
    return cubefft


def fft2d1d(cube):
    (nz,nx,ny) = np.shape(cube)
    cubefft = np.zeros_like(cube) + np.zeros_like(cube) * 1j                     # Convert to complex array
    for fm in np.arange(nz):
        cubefft[fm] = pfft.fft2(cube[fm])
    return cubefft

def ifft2d1d(cubefft):
    (nz,nx,ny) = np.shape(cubefft)
    cube = np.zeros_like(cubefft) + np.zeros_like(cubefft) * 1j                 # Convert to complex array
    for fm in np.arange(nz):
        cube[fm] = pfft.ifft2(cubefft[fm])
    return cube

#slow
def dec2d1dslow(cube,scale2d,scale1d,gen2=True,w2dname='starlet',w1dname='haar',wtype=1,normalization=False):

    (nz,nx,ny) = np.shape(cube)
    t0=time.time()
    band1d = np.zeros(scale1d+1)
    # 2d wavelet decomposition
    if w2dname == 'starlet':
        tmp = np.zeros((scale2d*nz,nx,ny))
        for fm in np.arange(nz):
            tmp[nz*(scale2d-1)+fm::-nz] = star2d(cube[fm],scale2d,gen2=gen2,normalization=normalization)     # Rerange the coefficients in the order of from low frequency to high frequency
    elif w2dname == 'direct':
        scale2d=1
        tmp = cube
        
    # 1d wavelet decomposition    
    for bd in np.arange(scale2d):
        for corx in np.arange(nx):
            for cory in np.arange(ny):
                (tmp2d1d,band1d) = wavOrth1d(tmp[bd*nz:(bd+1)*nz,corx,cory],scale1d,wname=w1dname,wtype=wtype)
                if bd == 0 and corx==0 and cory==0:
                    lenwt1d=len(tmp2d1d)
                    wt2d1d = np.zeros((scale2d*lenwt1d,nx,ny))

                wt2d1d[lenwt1d*bd:lenwt1d*(bd+1),corx,cory] = np.copy(tmp2d1d)
#	print "coucou 1"

    t1=time.time()
    #print "Time taken is "+str(t1-t0)

    return (wt2d1d,band1d)

#numba
def dec2d1d(cube,scale2d,scale1d,gen2=True,w2dname='starlet',w1dname='haar',wtype=1,normalization=False):
    #import time

    (nz,nx,ny) = np.shape(cube)

    t0=time.time()
    band1d = np.zeros(scale1d+1)
    # 2d wavelet decomposition
    if w2dname == 'starlet':
        tmp = np.zeros((scale2d*nz,nx,ny))
        for fm in np.arange(nz):
            tmp[nz*(scale2d-1)+fm::-nz] = star2d(cube[fm],scale2d,gen2=gen2,normalization=normalization)     # Rerange the coefficients in the order of from low frequency to high frequency
    elif w2dname == 'direct':
        scale2d=1
        tmp = cube

    #size=np.size(tmp)   
    #t1=time.time()
    #print "Time taken is "+str(t1-t0)
    #wt=np.array([])
    (test,bandtest) = wavOrth1d(tmp[0:nz,0,0],scale1d,wname=w1dname,wtype=wtype)
    lentest=len(test)
    size=len(tmp[0:nz,0,0])
    #print size,lentest
    #pdb.set_trace()
    #t2=time.time()
    ttt=make1Dtf(tmp,scale1d,scale2d,nx,ny,nz,size,lentest,w1dname)
    t3=time.time()
    #print "Time taken for make1Dtf = ",t3-t2
    #print "Time taken for dec2d1d = ",t3-t0
    #pdb.set_trace()
    return ttt #(wt2d1d,band1d) 

def custom_convolution(A,B,flag):
    dimA = A.shape[0]
    dimB = B.shape[0]
    dimC=dimA+dimB
    C=np.zeros(dimC,dtype=np.float64)
    for iA in np.arange(A.size):
        for iB in np.arange(B.size):
	    idx=iA+iB
            C[idx]+=A[iA]*B[iB]
    #print "In convolution"
    if flag:
        return C       # to reproduce 'full' mode of numpy.convolve
    else:
        return C[1:-1] # to reproduce 'valid' mode of numpy.convolve


fast_convolution = jit(double[:](double[:],double[:],int32))(custom_convolution)

@jit(nopython=False,parallel=False)
def make1Dtf(tmp,scale1d,scale2d,nx,ny,nz,size,lenwt1d,mode='haar'):
    # 1d wavelet decomposition
    scale = scale1d

    if mode == 'haar':
	print "Using Haar decomp"
        F = np.array([0.5,0.5])
        p = 1
        Lo_R = np.sqrt(2)*F/np.sum(F)
        Hi_R = Lo_R[::-1].copy()
        first = 2-p%2
        Hi_R[first::2] = -Hi_R[first::2]
	Hi_D=Hi_R[::-1].copy()
        Lo_D=Lo_R[::-1].copy()

    if mode == '9/7':
	print "Using 9/7 decomp"
        Df = np.array([0.0267487574110000,-0.0168641184430000,-0.0782232665290000,0.266864118443000,\
                           0.602949018236000,0.266864118443000,-0.0782232665290000,-0.0168641184430000,\
                           0.0267487574110000])
        Rf = np.array([-0.0456358815570000,-0.0287717631140000,0.295635881557000,0.557543526229000,\
                           0.295635881557000,-0.0287717631140000,-0.0456358815570000])

        lr = len(Rf)
        ld = len(Df)
        lmax = max(lr,ld)
        if lmax%2:
            lmax += 1
        Rf = np.hstack([np.zeros((lmax-lr)/2),Rf,np.zeros((lmax-lr+1)/2)])
        Df = np.hstack([np.zeros((lmax-ld)/2),Df,np.zeros((lmax-ld+1)/2)])
        
        p = 1
	first = 2-p%2
        Lo_R1 = np.sqrt(2)*Df/np.sum(Df)
        Hi_R1 = Lo_R1[::-1].copy()
        Hi_R1[first::2] = -Hi_R1[first::2]
        Hi_D1=Hi_R1[::-1].copy()
        Lo_D1=Lo_R1[::-1].copy()

        Lo_R2 = np.sqrt(2)*Rf/np.sum(Rf)
        Hi_R2 = Lo_R2[::-1].copy()
        Hi_R2[first::2] = -Hi_R2[first::2]
        Hi_D2=Hi_R2[::-1].copy()
        Lo_D2=Lo_R2[::-1].copy()
	Lo_D=Lo_D1
	Hi_D=Hi_D2


    (h0,g0) = (Lo_D,Hi_D)

    lf = h0.size 
    band = np.zeros(scale+1)
    band[-1] = size
    end = size
    start = 1
 
    wt2d1d = np.zeros((scale2d*lenwt1d,nx,ny),dtype=np.float64)
    #print wt2d1d.shape
    wt=np.array([],dtype=np.float64)
    for bd in prange(scale2d):
        for corx in prange(nx):
            for cory in prange(ny):
                x=tmp[bd*nz:(bd+1)*nz,corx,cory].copy()
		#pdb.set_trace()
                wt=np.array([],dtype=np.float64)
                #print "HELLO"      
                for sc in np.arange(scale-1):
 		    #print "Hello in scale",sc
            	    lsig = x.size
                    end = lsig + lf - 1
                    lenExt = lf - 1
		    #xExt=np.array([1,2,3],dtype=np.float64)
#                    xExt = np.lib.pad(x, (lenExt,lenExt), 'symmetric')
                    #xExt=np.concatenate((x[lenExt-1::-1],x,x[-lenExt+1:-lenExt-1:-1]),axis=0)
                    xExt=np.concatenate((x[0:lenExt][::-1],x,x[::-1][0:lenExt]),axis=0)
                    #print x
                    #print xExt
		    #pdb.set_trace()   
                    app = fast_convolution(xExt,h0,0) #np.convolve(xExt,h0,'valid')
                    x = app[start:end:2].copy()
                    detail = fast_convolution(xExt,g0,0) #np.convolve(xExt,g0,'valid')
                    #print app
                    #print detail
		    #pdb.set_trace()
		    #detail=np.array([1,2,3,4,5])
                    wt = np.hstack((detail[start:end:2],wt))     
                    band[-2-sc] = len(detail[start:end:2])
                    #print wt
		#print wt
                #pdb.set_trace()
                wt = np.hstack((x,wt)) 
                band[0] = len(x)
                #print "wt=",wt.shape
                #pdb.set_trace()
		index1=lenwt1d*bd
		index2=lenwt1d*(bd+1)
		a=wt.copy()
                wt2d1d[index1:index2,corx,cory] = a
		#print index1,index2,corx,cory,wt2d1d[index1:index2,corx,cory]
    #print "Before return in make1Dtf"

    return (wt2d1d,band)


###


###


@jit(nopython=False,parallel=False)
def make1Drec(wt,lenwt1d,band,scale2d,nx,ny,nz,mode='haar'):
   # 1d wavelet decomposition

    if mode == 'haar':
	print "Using Haar recomp"
        F = np.array([0.5,0.5])
        p = 1
        Lo_R = np.sqrt(2)*F/np.sum(F)
        Hi_R = Lo_R[::-1].copy()
        first = 2-p%2
        Hi_R[first::2] = -Hi_R[first::2]

    if mode == '9/7':
	print "Using 9/7 recomp"
        Df = np.array([0.0267487574110000,-0.0168641184430000,-0.0782232665290000,0.266864118443000,\
                           0.602949018236000,0.266864118443000,-0.0782232665290000,-0.0168641184430000,\
                           0.0267487574110000])
        Rf = np.array([-0.0456358815570000,-0.0287717631140000,0.295635881557000,0.557543526229000,\
                           0.295635881557000,-0.0287717631140000,-0.0456358815570000])

        lr = len(Rf)
        ld = len(Df)
        lmax = max(lr,ld)
        if lmax%2:
            lmax += 1
        Rf = np.hstack([np.zeros((lmax-lr)/2),Rf,np.zeros((lmax-lr+1)/2)])
        Df = np.hstack([np.zeros((lmax-ld)/2),Df,np.zeros((lmax-ld+1)/2)])
        
        p = 1
	first = 2-p%2
        Lo_R1 = np.sqrt(2)*Df/np.sum(Df)
        Hi_R1 = Lo_R1[::-1].copy()
        Hi_R1[first::2] = -Hi_R1[first::2]
        #Hi_D1=Hi_R1[::-1].copy()
        #Lo_D1=Lo_R1[::-1].copy()

        Lo_R2 = np.sqrt(2)*Rf/np.sum(Rf)
        #Hi_R2 = Lo_R2[::-1].copy()
        #Hi_R2[first::2] = -Hi_R2[first::2]
        #Hi_D2=Hi_R2[::-1].copy()
        #Lo_D2=Lo_R2[::-1].copy()
	Lo_R=Lo_R2
	Hi_R=Hi_R1
	

    (h1,g1) = (Lo_R,Hi_R)  #(h0,g0)=(Lo_D1,Hi_D2)
    tmp = np.zeros((nz*scale2d,nx,ny)) 
    for bd in prange(scale2d):
        for corx in prange(nx):
            for cory in prange(ny):
		tmpwt=wt[lenwt1d*bd:lenwt1d*(bd+1),corx,cory].copy()
    		sig = tmpwt[:np.int(band[0])].copy() #JGMOD
    		start = np.int(band[0]) #JGMOD     
                for sc in np.arange(np.size(band)-2):
            	    last = start+np.int(band[sc+1]) #JGMOD
                    detail = tmpwt[start:last].copy()
                    lsig = 2*sig.size
                    s = band[sc+2]
                    appInt = np.zeros(lsig-1)
                    appInt[::2] = sig.copy()
                    appInt = fast_convolution(appInt,h1,1)
                    first = np.int(np.floor(float(np.size(appInt) - s)/2.)) #JGMOD
                    last = np.int(np.size(appInt) - np.ceil(float(np.size(appInt) - s)/2.)) #JGMOD
                    appInt = appInt[first:last]            
                    detailInt = np.zeros(lsig-1)
                    detailInt[::2] = detail.copy()
                    detailInt = fast_convolution(detailInt,g1,1)
                    detailInt = detailInt[first:last]           
                    sig = appInt + detailInt 
                    start = last          
		tmp[bd*nz:(bd+1)*nz,corx,cory]=sig.copy()
    return tmp

def rec2d1d(wt,band1d,nz,gen2=True,adjoint=False,w2dname='starlet',w1dname='haar',wtype=1,normalization=False):
    t0=time.time()
    (nzwt,nx,ny) = np.shape(wt)
    lenwt1d=np.int(sum(band1d)-band1d[-1])

    if w2dname == 'starlet':
        scale2d = np.int(nzwt/lenwt1d) #JGMOD
    elif w2dname == 'direct':
        scale2d = 1
    #print "INFO *****"
    #print nx,ny,nz,scale2d
    #print nzwt,lenwt1d
    cube = np.zeros((nz,nx,ny))
    #tmp = np.zeros((nz*scale2d,nx,ny)) #tmp[bd*nz:(bd+1)*nz,corx,cory]
    # 1d reconstruction

    # Fast reconstruction
    tmp=make1Drec(wt,lenwt1d,band1d,scale2d,nx,ny,nz,w1dname)
  #  for bd in np.arange(scale2d):     
  #      for corx in np.arange(nx):
  #          for cory in np.arange(ny):
  #              tmp[bd*nz:(bd+1)*nz,corx,cory] = iwavOrth1d(wt[lenwt1d*bd:lenwt1d*(bd+1),corx,cory],band1d,wname=w1dname,wtype=wtype)
    # 2d reconstruction
    if w2dname=='starlet':
        for sc in np.arange(nz):
            if adjoint:
		indexx=(nz*(scale2d-1)+sc).astype(np.int32)
                cube[sc] = adstar2d(tmp[indexx::-nz],gen2=gen2,normalization=normalization)
            else:
		indexx=(nz*(scale2d-1)).astype(np.int32)
                cube[sc] = istar2d(tmp[indexx::-nz],gen2=gen2,normalization=normalization)
    elif w2dname=='direct':
        cube = tmp
    t3=time.time()
    #print "Time taken for rec2d1d = ",t3-t0
    return cube

# slow
def rec2d1dslow(wt,band1d,nz,gen2=True,adjoint=False,w2dname='starlet',w1dname='haar',wtype=1,normalization=False):
    t0=time.time()
    (nzwt,nx,ny) = np.shape(wt)
    lenwt1d=np.int(sum(band1d)-band1d[-1])
    if w2dname == 'starlet':
        scale2d = np.int(nzwt/lenwt1d) #JGMOD
    elif w2dname == 'direct':
        scale2d = 1
    #print "INFO *****"
    #print nx,ny,nz,scale2d
    #print nzwt,lenwt1d
    cube = np.zeros((nz,nx,ny))
    tmp = np.zeros((nz*scale2d,nx,ny))
    # 1d reconstruction
    for bd in np.arange(scale2d):     
        for corx in np.arange(nx):
            for cory in np.arange(ny):
                tmp[bd*nz:(bd+1)*nz,corx,cory] = iwavOrth1d(wt[lenwt1d*bd:lenwt1d*(bd+1),corx,cory],band1d,wname=w1dname,wtype=wtype)
    # 2d reconstruction
    if w2dname=='starlet':
        for sc in np.arange(nz):
            if adjoint:
		indexx=(nz*(scale2d-1)+sc).astype(np.int32)
                cube[sc] = adstar2d(tmp[indexx::-nz],gen2=gen2,normalization=normalization)
            else:
		indexx=(nz*(scale2d-1)).astype(np.int32)
                cube[sc] = istar2d(tmp[indexx::-nz],gen2=gen2,normalization=normalization)
    elif w2dname=='direct':
        cube = tmp
    t3=time.time()
    #print "Time taken for rec2d1d = ",t3-t0
    return cube

# def dec2d1d_boost(cube,mr2d,mr1d,gen2=True,normalization=False):
#     (nz,nx,ny) = np.shape(cube)
#     scale2d = mr2d.nbr_scale()
#     scale1d = mr1d.nbr_scale()
#     wt2d1d = np.zeros((scale2d*nz,nx,ny))
#     band1d = np.zeros(scale1d+1)
#     tmp = np.zeros((scale2d*nz,nx,ny))
#     # 2d wavelet decomposition
#     for fm in np.arange(nz):
#         tmp[nz*(scale2d-1)+fm::-nz] = star2d_boost(cube[fm],mr2d,gen2,normalization)                         # Rerange the coefficients in the order of from low frequency to high frequency
#     # 1d wavelet decomposition    
#     for bd in np.arange(scale2d):
#         for corx in np.arange(nx):
#             for cory in np.arange(ny):
#                 (wt2d1d[bd*nz:(bd+1)*nz,corx,cory],band1d)=wavOrth1d_boost(tmp[bd*nz:(bd+1)*nz,corx,cory],mr1d)
#     return (wt2d1d,band1d)
# 
# def rec2d1d_boost(wt,band1d,nz,mr2d,mr1d,gen2=True,adjoint=False,normalization=False):
#     scale2d = mr2d.nbr_scale()
#     (nzwt,nx,ny) = np.shape(wt)
#     nz = nzwt/scale2d
#     cube = np.zeros((nz,nx,ny))
#     tmp = np.zeros((nzwt,nx,ny))
#     # 1d reconstruction
#     for bd in np.arange(scale2d):     
#         for corx in np.arange(nx):
#             for cory in np.arange(ny):
#                 tmp[bd*nz:(bd+1)*nz,corx,cory] = iwavOrth1d_boost(wt[bd*nz:(bd+1)*nz,corx,cory],mr1d)
#     # 2d reconstruction
#     for sc in np.arange(nz):
#         cube[sc] = istar2d_boost(tmp[nz*(scale2d-1)+sc::-nz],mr2d,gen2,adjoint,normalization)
#     
#     return cube

# def applyMSpositivity(p,scale2d,Piter):
#     eps = 0.001
#     (nz,nx,ny) = np.shape(p)
#     pp = np.zeros_like(p)
#     stWav = sparse2d.Starlet2D(nx,ny,scale2d)
#     for fm in np.arange(nz):
#         pWav = stWav.transform(p[fm],True)
#         mwt = pWav.max()/2
#         rsdPos = np.copy(p[fm])
#         pWav = np.zeros((scale2d,nx,ny))
#         for it in np.arange(Piter+5):
#             ld = mwt - (mwt - eps)*it/(Piter-1)
#             if ld < 0:
#                 ld = 0
#             tmp = stWav.transform(rsdPos,True)
#             pWav += tmp
#             softTh(pWav,ld)
#             pWav[pWav<eps] = 0
#             rec = stWav.reconstruct(pWav,True)
#             rsdPos = p[fm] - rec
#         pp[fm] = np.copy(rec)
#     return pp

# def updateMultiResol(mrSup,thTab,alpha,alpha_sh):
#     (nz,nx,ny) = np.shape(thTab)
#     for sc in np.arange(nz):
#         mrSup[np.abs(alpha-alpha_sh)<]
        
    
